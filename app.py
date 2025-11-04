# app.py — FINAL v2.2c (dirs auto-heal, PERSIST_DIR/PERSISTENT_DIR env)
# (train→predict→next-group 파이프라인, 부팅시 필수 경로/빈 로그 보장, 예측락 stale GC, 그룹학습 락/게이트)

from flask import Flask, jsonify, request, Response
from recommend import main
import train, os, threading, datetime, pytz, traceback, sys, shutil, re, time
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_trigger import run as trigger_run
from predict_trigger import _is_group_complete_for_all_strategies, _get_current_group_symbols
from data.utils import SYMBOLS, get_kline_by_strategy
from data.utils import (
    ready_for_group_predict, mark_group_predicted, group_all_complete,
    get_current_group_symbols, SYMBOL_GROUPS
)
from visualization import generate_visual_report, generate_visuals_for_strategy
from wrong_data_loader import load_training_prediction_data
from predict import evaluate_predictions
from train import train_symbol_group_loop  # compatibility
import maintenance_fix_meta
from logger import ensure_prediction_log_exists, ensure_train_log_exists, PREDICTION_HEADERS, TRAIN_HEADERS
from logger import log_audit_prediction as log_audit
from config import get_TRAIN_LOG_PATH

# === 공통 경로/디렉토리 ===
# NOTE: Render에서는 /persistent 쓰면 Permission denied가 뜨므로 기본값을 /tmp/persistent 로 둔다.
#       로컬/자체 서버에서 예전처럼 /persistent 쓰고 싶으면
#       PERSIST_DIR=/persistent 또는 PERSISTENT_DIR=/persistent 로 환경변수만 주면 된다.
PERSIST_DIR = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/tmp/persistent"
)
LOG_DIR     = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR   = os.path.join(PERSIST_DIR, "models")
RUN_DIR     = os.path.join(PERSIST_DIR, "run")

# --- ✨ 필수 폴더 자동 생성 유틸 (부팅/리셋/조기QWIPE 후 재사용) ---
NEEDED_DIRS = [
    f"{PERSIST_DIR}/importances",
    f"{PERSIST_DIR}/guanwu/incoming",
    LOG_DIR,
    MODEL_DIR,
    RUN_DIR,
    "/tmp/importances",
]
def ensure_dirs():
    for p in NEEDED_DIRS:
        os.makedirs(p, exist_ok=True)

# 모듈 로드 시 1회 보장
ensure_dirs()

# integrity guard optional
try:
    from integrity_guard import run as _integrity_check
    _integrity_check()
except Exception as e:
    print(f"[WARN] integrity_guard skipped: {e}")

from diag_e2e import run as diag_e2e_run

# cleanup modules
try:
    from scheduler_cleanup import start_cleanup_scheduler
    import safe_cleanup
    import scheduler_cleanup as _cleanup_mod
except Exception:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scheduler_cleanup import start_cleanup_scheduler
        import safe_cleanup
        import scheduler_cleanup as _cleanup_mod
    except Exception:
        start_cleanup_scheduler = lambda: None
        safe_cleanup = type("sc", (), {"get_directory_size_gb": lambda p: 0, "HARD_CAP_GB": 9.6, "run_emergency_purge": lambda: None, "cleanup_logs_and_models": lambda: None})
        _cleanup_mod = safe_cleanup

# predict-lock stale GC
try:
    import predict_lock as _pl
    _pl_clear = getattr(_pl, "clear_stale_predict_lock", lambda: None)
except Exception:
    _pl = None
    _pl_clear = lambda: None

# predict gate
try:
    from predict import open_predict_gate, close_predict_gate, predict
except Exception:
    def open_predict_gate(*args, **kwargs): return None
    def close_predict_gate(*args, **kwargs): return None
    def predict(*args, **kwargs): raise RuntimeError("predict 불가")

def _safe_open_gate(note: str = ""):
    try:
        open_predict_gate(note=note)
        print(f"[gate] open ({note})"); sys.stdout.flush()
    except Exception as e:
        print(f"[gate] open err: {e}"); sys.stdout.flush()

def _safe_close_gate(note: str = ""):
    try:
        close_predict_gate(note=note)
        print(f"[gate] close ({note})"); sys.stdout.flush()
    except Exception as e:
        print(f"[gate] close err: {e}"); sys.stdout.flush()

# [ADD] 그룹잠금 전용 파일
GROUP_TRAIN_LOCK = os.path.join(RUN_DIR, "group_training.lock")

DEPLOY_ID  = os.getenv("RENDER_RELEASE_ID") or os.getenv("RENDER_GIT_COMMIT") or os.getenv("RENDER_SERVICE_ID") or "local"
BOOT_MARK  = os.path.join(PERSIST_DIR, f".boot_notice_{DEPLOY_ID}")

# locks
LOCK_DIR   = getattr(safe_cleanup, "LOCK_DIR", os.path.join(PERSIST_DIR, "locks"))
LOCK_PATH  = getattr(safe_cleanup, "LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))
os.makedirs(LOCK_DIR, exist_ok=True)

# === GROUP_ACTIVE 마커 경로 ===
GROUP_ACTIVE_PATH = os.path.join(PERSIST_DIR, "GROUP_ACTIVE")

def _set_group_active(active: bool, group_idx: int | None = None, symbols: list | None = None):
    try:
        if active:
            with open(GROUP_ACTIVE_PATH, "w", encoding="utf-8") as f:
                ts = datetime.datetime.utcnow().isoformat()
                syms = ",".join(symbols or [])
                f.write(f"ts={ts}\n")
                if group_idx is not None:
                    f.write(f"group={group_idx}\n")
                f.write(f"symbols={syms}\n")
            print(f"[GROUP_ACTIVE] set group={group_idx} symbols={symbols}"); sys.stdout.flush()
        else:
            if os.path.exists(GROUP_ACTIVE_PATH):
                os.remove(GROUP_ACTIVE_PATH)
                print("[GROUP_ACTIVE] cleared"); sys.stdout.flush()
    except Exception as e:
        print(f"[GROUP_ACTIVE] set/clear err: {e}"); sys.stdout.flush()

def _is_group_active_file() -> bool:
    try:
        return os.path.exists(GROUP_ACTIVE_PATH)
    except Exception:
        return False

# BOOT: orphan 전역락 제거 + 예측락 stale GC + 그룹학습락 제거
if os.path.exists(LOCK_PATH):
    try:
        os.remove(LOCK_PATH)
        print("[BOOT] orphan lock removed"); sys.stdout.flush()
    except Exception as e:
        print(f"[BOOT] lock remove failed: {e}"); sys.stdout.flush()
try:
    _pl_clear(); print("[BOOT] predict lock stale-GC done")
except Exception as e:
    print(f"[BOOT] predict lock GC failed: {e}")
try:
    if os.path.exists(GROUP_TRAIN_LOCK):
        os.remove(GROUP_TRAIN_LOCK)
        print("[BOOT] stale GROUP_TRAIN_LOCK removed"); sys.stdout.flush()
except Exception as e:
    print(f"[BOOT] GROUP_TRAIN_LOCK cleanup failed: {e}"); sys.stdout.flush()

def _acquire_global_lock():
    try:
        os.makedirs(LOCK_DIR, exist_ok=True)
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(f"locked at {datetime.datetime.now().isoformat()}\n")
        print(f"[LOCK] created: {LOCK_PATH}"); sys.stdout.flush()
        return True
    except Exception as e:
        print(f"[LOCK] create failed: {e}"); sys.stdout.flush()
        return False

def _release_global_lock():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
            print(f"[LOCK] removed: {LOCK_PATH}"); sys.stdout.flush()
            return True
    except Exception as e:
        print(f"[LOCK] remove failed: {e}"); sys.stdout.flush()
    return False

# ===== train API safe wrappers =====
def _is_training() -> bool:
    try:
        return bool(getattr(train, "is_loop_running", lambda: False)())
    except Exception:
        return False

def _start_train_loop_safe(force_restart=False, sleep_sec=0):
    fn = getattr(train, "start_train_loop", None)
    if callable(fn):
        try:
            return bool(fn(force_restart=force_restart, sleep_sec=sleep_sec))
        except Exception:
            try:
                fn(force_restart=force_restart, sleep_sec=sleep_sec)
                return True
            except Exception:
                return False
    for name in ("start_train_loop", "start_loop", "start"):
        fn2 = getattr(train, name, None)
        if callable(fn2):
            try:
                fn2()
                return True
            except Exception:
                continue
    return False

def _stop_train_loop_safe(timeout=30):
    fn = getattr(train, "stop_train_loop", None)
    if callable(fn):
        try:
            return bool(fn(timeout=timeout))
        except Exception:
            try:
                fn()
                return True
            except Exception:
                return False
    fn2 = getattr(train, "request_stop", None)
    if callable(fn2):
        try:
            fn2()
            return True
        except Exception:
            pass
    return False

def _request_stop_safe():
    fn = getattr(train, "request_stop", None)
    if callable(fn):
        try:
            fn()
            return True
        except Exception:
            return False
    return False

def _train_models_safe(symbols):
    fn = getattr(train, "train_models", None)
    if callable(fn):
        try:
            fn(symbols)
            return True
        except Exception as e:
            print(f"[TRAIN] train_models failed: {e}")
            return False
    fn2 = getattr(train, "train_symbol_group_loop", None)
    if callable(fn2):
        try:
            fn2(symbols)
            return True
        except Exception:
            return False
    return False

def _await_models_visible(symbols, timeout_sec=20, poll_sec=0.5):
    fn = getattr(train, "_await_models_visible", None)
    if callable(fn):
        try:
            return fn(symbols, timeout_sec=timeout_sec)
        except Exception:
            pass
    exts = (".pt", ".ptz", ".safetensors")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        found = set()
        try:
            if os.path.isdir(MODEL_DIR):
                for f in os.listdir(MODEL_DIR):
                    for s in symbols:
                        if f.startswith(f"{s}_") and f.endswith(exts):
                            found.add(s)
                for s in symbols:
                    sdir = os.path.join(MODEL_DIR, s)
                    if os.path.isdir(sdir):
                        for strat in ("단기", "중기", "장기"):
                            d = os.path.join(sdir, strat)
                            if os.path.isdir(d):
                                if any(name.endswith(exts) for name in os.listdir(d)):
                                    found.add(s)
        except Exception:
            pass
        if found:
            return sorted(found)
        time.sleep(poll_sec)
    return []

def _has_model_for(symbol, strategy):
    fn = getattr(train, "_has_model_for", None)
    if callable(fn):
        try:
            return bool(fn(symbol, strategy))
        except Exception:
            pass
    try:
        exts = (".pt", ".ptz", ".safetensors")
        pref = f"{symbol}_{strategy}_"
        if os.path.isdir(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.startswith(pref) and f.endswith(exts):
                    return True
            d = os.path.join(MODEL_DIR, symbol, strategy)
            if os.path.isdir(d):
                if any(name.endswith(exts) for name in os.listdir(d)):
                    return True
    except Exception:
        pass
    return False

# quarantine wipe helper
def _quarantine_wipe_persistent():
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trash_dir = os.path.join(PERSIST_DIR, f"_trash_{ts}")
    os.makedirs(trash_dir, exist_ok=True)
    keep_names = {os.path.basename(LOCK_DIR)}
    moved = []
    for name in list(os.listdir(PERSIST_DIR)):
        if name in keep_names: continue
        src = os.path.join(PERSIST_DIR, name)
        dst = os.path.join(trash_dir, name)
        try:
            shutil.move(src, dst); moved.append(name)
        except Exception as e:
            print(f"⚠️ [QWIPE] move 실패: {src} -> {dst} ({e})")
    # ✅ 핵심: 바로 재생성(학습/특징중요도 저장 중에도 디렉터리 존재 보장)
    ensure_dirs()
    print(f"🧨 [QWIPE] moved_to_trash={moved} trash_dir={trash_dir}"); sys.stdout.flush()
    return trash_dir

def _async_emergency_purge():
    try:
        try:
            for name in list(os.listdir(PERSIST_DIR)):
                if name.startswith("_trash_"):
                    path = os.path.join(PERSIST_DIR, name)
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"[BOOT-CLEANUP] trashed removed: {name}")
        except Exception as e:
            print(f"⚠️ [BOOT-CLEANUP] trash 제거 실패: {e}")
        used_gb = safe_cleanup.get_directory_size_gb(PERSIST_DIR)
        hard_cap = getattr(safe_cleanup, "HARD_CAP_GB", 9.6)
        print(f"[BOOT-CLEANUP] used={used_gb:.2f}GB hard_cap={hard_cap:.2f}GB"); sys.stdout.flush()
        if used_gb >= hard_cap:
            print("[EMERGENCY] pre-DB purge 시작 (하드캡 초과)"); sys.stdout.flush()
            safe_cleanup.run_emergency_purge()
            print("[EMERGENCY] pre-DB purge 완료"); sys.stdout.flush()
        else:
            if os.getenv("CLEANUP_ON_BOOT", "0") == "1":
                print("[BOOT-CLEANUP] CLEANUP_ON_BOOT=1 → 온건 정리 실행"); sys.stdout.flush()
                safe_cleanup.cleanup_logs_and_models()
                print("[BOOT-CLEANUP] 완료"); sys.stdout.flush()
            else:
                print("[BOOT-CLEANUP] 비활성화(CLEANUP_ON_BOOT=0)"); sys.stdout.flush()
    except Exception as e:
        print(f"[경고] pre-DB purge/cleanup 결정 실패: {e}"); sys.stdout.flush()

threading.Thread(target=_async_emergency_purge, daemon=True).start()

PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
LOG_FILE          = get_TRAIN_LOG_PATH()
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG         = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG       = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_LOG       = os.path.join(LOG_DIR, "failure_count.csv")

# ensure logs
try:
    ensure_train_log_exists()
except Exception:
    pass
ensure_prediction_log_exists()

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# 그룹 완주 후에만 예측 허용
REQUIRE_GROUP_COMPLETE = int(os.getenv("REQUIRE_GROUP_COMPLETE", "1"))

# scheduler
_sched = None
def start_scheduler():
    global _sched
    if _sched is not None:
        print("⚠️ 스케줄러 이미 실행 중, 재시작 생략"); sys.stdout.flush()
        return
    if os.path.exists(LOCK_PATH):
        print("⏸️ 리셋 락 감지 → 스케줄러 시작 지연"); sys.stdout.flush()
        def _deferred():
            backoff = 1.0
            while os.path.exists(LOCK_PATH):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
            try:
                start_scheduler(); print("▶️ 지연 후 스케줄러 시작"); sys.stdout.flush()
            except Exception as e:
                print(f"❌ 지연 시작 실패: {e}")
        threading.Thread(target=_deferred, daemon=True).start()
        return

    try:
        _pl_clear(); print("[SCHED] predict lock stale-GC pre-start")
    except Exception as e:
        print(f"[SCHED] predict lock GC failed pre-start: {e}")

    print(">>> 스케줄러 시작"); sys.stdout.flush()
    sched = BackgroundScheduler(
        timezone=pytz.timezone("Asia/Seoul"),
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 90},
    )

    def 평가작업(strategy):
        def wrapped():
            try:
                if _is_training() or os.path.exists(LOCK_PATH) or _is_group_active_file() or os.path.exists(GROUP_TRAIN_LOCK):
                    print(f"[EVAL] skip: training/lock/group-active (strategy={strategy})"); sys.stdout.flush()
                    return
                ts = now_kst().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[EVAL][{ts}] 전략={strategy} 시작"); sys.stdout.flush()
                evaluate_predictions(lambda sym, _: get_kline_by_strategy(sym, strategy))
            except Exception as e:
                print(f"[EVAL] {strategy} 실패: {e}")
        return wrapped

    for strat in ["단기", "중기", "장기"]:
        sched.add_job(
            평가작업(strat),
            trigger="interval",
            minutes=30,
            id=f"eval_{strat}",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=90,
        )

    def _pred_lock_gc():
        try:
            _pl_clear()
        except Exception as e:
            print(f"[LOCK] periodic GC fail: {e}")
    sched.add_job(
        _pred_lock_gc,
        "interval",
        minutes=int(os.getenv("PREDICT_LOCK_GC_MIN", "5")),
        id="predict_lock_gc",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=90,
    )

    def _predict_job():
        try:
            if _is_training() or os.path.exists(LOCK_PATH) or _is_group_active_file() or os.path.exists(GROUP_TRAIN_LOCK):
                print("[PREDICT] skip: training/lock/group-active"); sys.stdout.flush()
                return
            _pl_clear()
            print("[PREDICT] trigger_run start"); sys.stdout.flush()
            _safe_open_gate("sched_trigger")
            try:
                trigger_run()
            except Exception as e:
                print(f"[PREDICT] ❌ trigger_run 실패: {e}")
                try:
                    from predict import failed_result
                    failed_result("ALL", "auto", reason=str(e), source="sched_trigger")
                except Exception:
                    pass
            finally:
                _safe_close_gate("sched_trigger")
            print("[PREDICT] trigger_run done"); sys.stdout.flush()
        except Exception as e:
            print(f"[PREDICT] 실패: {e}")

    sched.add_job(
        _predict_job,
        "interval",
        minutes=30,
        id="predict_trigger",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=90,
    )

    def meta_fix_job():
        try:
            maintenance_fix_meta.fix_all_meta_json()
        except Exception as e:
            print(f"[META-FIX] 주기작업 실패: {e}")

    sched.add_job(
        meta_fix_job,
        "interval",
        minutes=30,
        id="meta_fix",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=90,
    )

    sched.start()
    _sched = sched
    print("✅ 스케줄러 시작 완료"); sys.stdout.flush()

def _pause_and_clear_scheduler():
    global _sched
    try:
        if _sched is not None:
            print("[SCHED] pause + remove_all_jobs"); sys.stdout.flush()
            try: _sched.pause()
            except Exception: pass
            try: _sched.remove_all_jobs()
            except Exception: pass
            try: _sched.shutdown(wait=False)
            except Exception: pass
            _sched = None
    except Exception as e:
        print(f"[SCHED] 정지 실패: {e}"); sys.stdout.flush()

def _stop_all_aux_schedulers():
    try:
        _pause_and_clear_scheduler()
    except Exception:
        pass
    try:
        if hasattr(_cleanup_mod, "stop_cleanup_scheduler"):
            try:
                _cleanup_mod.stop_cleanup_scheduler()
                print("🧹 [SCHED] cleanup 스케줄러 stop 호출"); sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ cleanup stop 실패: {e}"); sys.stdout.flush()
        for name in dir(_cleanup_mod):
            obj = getattr(_cleanup_mod, name, None)
            if isinstance(obj, BackgroundScheduler):
                try:
                    obj.shutdown(wait=False)
                    print(f"🧹 [SCHED] cleanup.{name} shutdown"); sys.stdout.flush()
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ cleanup 스케줄러 탐지 실패: {e}"); sys.stdout.flush()

# Flask app
app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

# init once
_INIT_DONE = False
_INIT_LOCK = threading.Lock()

def _init_background_once():
    global _INIT_DONE
    with _INIT_LOCK:
        if _INIT_DONE:
            return
        try:
            if os.path.exists(LOCK_PATH):
                print("⏸️ 락 감지 → 백그라운드 초기화 지연"); sys.stdout.flush()
                return
            from failure_db import ensure_failure_db
            print(">>> 서버 실행 준비")
            ensure_failure_db(); print("✅ failure_patterns DB 초기화 완료")

            # 필수 경로 및 빈 로그 보장(폴더는 이미 생성됨)
            try: ensure_train_log_exists()
            except Exception: pass
            try: ensure_prediction_log_exists()
            except Exception: pass

            _pl_clear()
            print("[pipeline] serialized: train -> predict -> next-group"); sys.stdout.flush()

            autostart = os.getenv("APP_AUTOSTART_TRAIN", "1") != "0"
            _safe_close_gate("init_train_start")
            if autostart:
                started = _start_train_loop_safe(force_restart=False, sleep_sec=0)
                print("✅ 학습 루프 스레드 시작" if started else "⚠️ 학습 루프 시작 실패 또는 이미 실행 중")
            else:
                print("⏸️ 학습 루프 자동 시작 안함")

            start_cleanup_scheduler()
            print("✅ cleanup 스케줄러 시작")
            try:
                start_scheduler()
            except Exception as e:
                print(f"⚠️ 스케줄러 시작 실패: {e}")

            threading.Thread(target=maintenance_fix_meta.fix_all_meta_json, daemon=True).start()
            print("✅ maintenance_fix_meta 초기 실행 트리거")

            try:
                fd = os.open(BOOT_MARK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                send_message("[시작] YOPO 서버 실행됨")
                print("✅ Telegram 알림 발송 완료")
            except FileExistsError:
                print("ℹ️ 부팅 알림 생략")
            except Exception as e:
                print(f"⚠️ Telegram 발송 실패: {e}")

            _INIT_DONE = True
            print("✅ 백그라운드 초기화 완료")
        except Exception as e:
            print(f"❌ 백그라운드 초기화 실패: {e}")

if hasattr(app, "before_serving"):
    @app.before_serving
    def _boot_once():
        _init_background_once()
else:
    @app.before_request
    def _boot_once_compat():
        if not _INIT_DONE:
            _init_background_once()

# 학습 직후 예측: 그룹 완주 가드
def _predict_after_training(symbols, source_note):
    if not symbols:
        return
    try:
        if REQUIRE_GROUP_COMPLETE == 1:
            group_syms = _get_current_group_symbols()
            if isinstance(group_syms, (list, tuple)) and len(group_syms) > 0:
                if not _is_group_complete_for_all_strategies(list(group_syms)):
                    print(f"[APP-PRED] 🚫 그룹 미완료 → 예측 차단 ({source_note})"); sys.stdout.flush()
                    for sym in sorted(set(symbols)):
                        try: log_audit(sym, "ALL", "학습후예측스킵", "그룹 미완료(REQUIRE_GROUP_COMPLETE=1)")
                        except Exception: pass
                    return
    except Exception as e:
        print(f"[APP-PRED] 그룹완료검증 실패: {e}")

    try:
        await_sec = int(os.getenv("PREDICT_MODEL_AWAIT_SEC","60"))
    except Exception:
        await_sec = 60
    vis = _await_models_visible(symbols, timeout_sec=await_sec)
    if not vis:
        print(f"[APP-PRED] 모델 가시화 실패 → 예측 생략 candidates={sorted(set(symbols))}")
        return
    if os.path.exists(LOCK_PATH):
        try:
            os.remove(LOCK_PATH)
            print("[APP-PRED] cleared stale lock before predict"); sys.stdout.flush()
        except Exception as e:
            print(f"[APP-PRED] lock remove failed: {e}"); sys.stdout.flush()
    _pl_clear()
    _safe_open_gate(source_note)
    try:
        for sym in sorted(set(vis)):
            for strat in ["단기","중기","장기"]:
                try:
                    if not _has_model_for(sym, strat):
                        print(f"[APP-PRED] skip {sym}-{strat}: model missing")
                        continue
                    print(f"[APP-PRED] predict {sym}-{strat}")
                    try:
                        result = predict(sym, strat, source=source_note, model_type=None)
                        if not isinstance(result, dict):
                            print(f"[APP-PRED] ⚠️ invalid return"); sys.stdout.flush()
                            try:
                                from predict import failed_result
                                failed_result(sym, strat, reason="invalid_return", source="app_predict")
                            except Exception: pass
                        else:
                            print(f"[APP-PRED] ✅ {sym}-{strat} ok: {result.get('reason','ok')}"); sys.stdout.flush()
                    except Exception as e:
                        print(f"[APP-PRED] ❌ 예측 오류: {e}"); sys.stdout.flush()
                        try:
                            from predict import failed_result
                            failed_result(sym, strat, reason=str(e), source="app_predict")
                        except Exception: pass
                except Exception as e:
                    print(f"[APP-PRED] {sym}-{strat} 실패: {e}"); sys.stdout.flush()
    finally:
        _safe_close_gate(source_note + "_end")

# routes
@app.route("/")
def index(): return "Yopo server is running"

@app.route("/ping")
def ping(): return "pong"

@app.route("/admin/clear-predict-lock", methods=["POST","GET"])
def clear_predict_lock_admin():
    try:
        _pl_clear()
        return "✅ predict lock stale-GC executed"
    except Exception as e:
        return f"⚠️ fail: {e}", 500

@app.route("/yopo-health")
def yopo_health():
    logs, strategy_html, problems = {}, [], []
    file_map = {"pred": PREDICTION_LOG, "train": LOG_FILE, "audit": AUDIT_LOG, "msg": MESSAGE_LOG}
    for name, path in file_map.items():
        try:
            logs[name] = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
            if "timestamp" in logs[name].columns:
                logs[name] = logs[name][logs[name]["timestamp"].notna()]
        except Exception:
            logs[name] = pd.DataFrame()
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith((".pt", ".ptz", ".safetensors"))]
    except Exception:
        model_files = []
    model_info = {}
    for f in model_files:
        m = re.match(r"(.+?)_(단기|중기|장기)_(lstm|cnn_lstm|transformer)(?:_.*)?\.(pt|ptz|safetensors)$", f)
        if m:
            symbol, strat, mtype, _ext = m.groups()
            model_info.setdefault(strat, {}).setdefault(symbol, set()).add(mtype)
    for strat in ["단기", "중기", "장기"]:
        try:
            pred  = logs.get("pred",  pd.DataFrame())
            train_log_df = logs.get("train", pd.DataFrame())
            audit = logs.get("audit", pd.DataFrame())
            pred  = pred.query(f"strategy == '{strat}'")  if not pred.empty  else pd.DataFrame()
            train_log_q = train_log_df.query(f"strategy == '{strat}'") if not train_log_df.empty else pd.DataFrame()
            audit = audit.query(f"strategy == '{strat}'") if not audit.empty else pd.DataFrame()
            if not pred.empty and "status" in pred.columns:
                pred["volatility"] = pred["status"].astype(str).str.startswith("v_")
            else:
                pred["volatility"] = False
            try:
                pred["return"] = pd.to_numeric(pred.get("return_value", pred.get("rate", pd.Series(dtype=float))), errors="coerce").fillna(0.0)
            except Exception:
                pred["return"] = 0.0
            nvol = pred[~pred["volatility"]] if not pred.empty else pd.DataFrame()
            vol  = pred[ pred["volatility"]] if not pred.empty else pd.DataFrame()
            def stat(df, s):
                try:
                    return int(((not df.empty) and ("status" in df.columns)) and (df["status"] == s).sum()) or 0
                except Exception:
                    return 0
            sn, fn, pn_, fnl = map(lambda s: stat(nvol, s), ["success", "fail", "pending", "failed"])
            sv, fv, pv, fvl = map(lambda s: stat(vol,  s), ["v_success", "v_fail", "pending", "failed"])
            def perf(df, kind="일반"):
                try:
                    s = stat(df, "v_success" if kind == "변동성" else "success")
                    f = stat(df, "v_fail"    if kind == "변동성" else "fail")
                    t = s + f
                    avg = float(df["return"].mean()) if ("return" in df) and (df.shape[0] > 0) else 0.0
                    return {"succ": s, "fail": f, "succ_rate": s/t*100 if t else 0,
                            "fail_rate": f/t*100 if t else 0, "r_avg": avg, "total": t}
                except Exception:
                    return {"succ": 0, "fail": 0, "succ_rate": 0, "fail_rate": 0, "r_avg": 0, "total": 0}
            pn, pv_stats = perf(nvol, "일반"), perf(vol, "변동성")
            strat_models = model_info.get(strat, {})
            types = {"lstm": 0, "cnn_lstm": 0, "transformer": 0}
            for mtypes in strat_models.values():
                for t in mtypes: types[t] += 1
            trained_syms = [s for s, t in strat_models.items() if {"lstm","cnn_lstm","transformer"}.issubset(t)]
            try:
                untrained = sorted(set(SYMBOLS) - set(trained_syms))
            except Exception:
                untrained = []
            if sum(types.values()) == 0: problems.append(f"{strat}: 모델 없음")
            if sn + fn + pn_ + fnl + sv + fv + pv + fvl == 0: problems.append(f"{strat}: 예측 없음")
            if pn["total"] == 0: problems.append(f"{strat}: 평가 미작동")
            if pn["fail_rate"]  > 50: problems.append(f"{strat}: 일반 실패율 {pn['fail_rate']:.1f}%")
            if pv_stats["fail_rate"] > 50: problems.append(f"{strat}: 변동성 실패율 {pv_stats['fail_rate']:.1f}%")
            table = "<i style='color:gray'>최근 예측 없음 또는 컬럼 부족</i>"
            required_cols = {"timestamp","symbol","strategy","direction","return","status"}
            if (pred.shape[0] > 0) and required_cols.issubset(set(pred.columns)):
                recent10 = pred.sort_values("timestamp").tail(10).copy()
                rows = []
                for _, r in recent10.iterrows():
                    rtn = r.get("return", 0.0) or r.get("rate", 0.0)
                    try: rtn_pct = f"{float(rtn) * 100:.2f}%"
                    except: rtn_pct = "0.00%"
                    s = str(r.get('status',''))
                    status_icon = '✅' if s in ['success','v_success'] else '❌' if s in ['fail','v_fail'] else '⏳' if s in ['pending','v_pending'] else '🛑'
                    rows.append(f"<tr><td>{r.get('timestamp','')}</td><td>{r.get('symbol','')}</td><td>{r.get('direction','')}</td><td>{rtn_pct}</td><td>{status_icon}</td></tr>")
                table = "<table border='1' style='margin-top:4px'><tr><th>시각</th><th>심볼</th><th>방향</th><th>수익률</th><th>상태</th></tr>" + "".join(rows) + "</table>"
            last_train = train_log_q['timestamp'].iloc[-1] if (not train_log_q.empty and 'timestamp' in train_log_q) else '없음'
            last_pred  = pred['timestamp'].iloc[-1]  if (not pred.empty and 'timestamp' in pred)  else '없음'
            last_audit = audit['timestamp'].iloc[-1] if (not audit.empty and 'timestamp' in audit) else '없음'
            info_html = f"""<div style='border:1px solid #aaa;margin:16px 0;padding:10px;font-family:monospace;background:#f8f8f8;'>
<b style='font-size:16px;'>📌 전략: {strat}</b><br>
- 모델 수: {sum(types.values())} (lstm={types['lstm']}, cnn={types['cnn_lstm']}, trans={types['transformer']})<br>
- 심볼 수: {len(SYMBOLS)} | 완전학습: {len(trained_syms)} | 미완성: {len(untrained)}<br>
- 최근 학습: {last_train}<br>
- 최근 예측: {last_pred}<br>
- 최근 평가: {last_audit}<br>
- 예측 (일반): {sn + fn + pn_ + fnl}건 (✅{sn} ❌{fn} ⏳{pn_} 🛑{fnl})<br>
- 예측 (변동성): {sv + fv + pv + fvl}건 (✅{sv} ❌{fv} ⏳{pv} 🛑{fvl})<br>
<b style='color:#000088'>🎯 일반 예측</b>: {pn['total']}건 | {pn['succ_rate']:.1f}% / {pn['fail_rate']:.1f}% / {pn['r_avg']:.2f}%<br>
<b style='color:#880000'>🌪️ 변동성 예측</b>: {pv_stats['total']}건 | {pv_stats['succ_rate']:.1f}% / {pv_stats['fail_rate']:.1f}% / {pv_stats['r_avg']:.2f}%<br>
<b>📋 최근 예측 10건</b><br>{table}
</div>"""
            try:
                visual = generate_visuals_for_strategy(strat)
            except Exception as e:
                visual = f"<div style='color:red'>[시각화 실패: {e}]</div>"
            strategy_html.append(f"<div style='margin-bottom:30px'>{info_html}<div style='margin:20px 0'>{visual}</div><hr></div>")
        except Exception as e:
            strategy_html.append(f"<div style='color:red;'>❌ {strat} 실패: {type(e).__name__} → {e}</div>")
    status = "🟢 전체 전략 정상 작동 중" if not problems else "🔴 종합진단 요약:<br>" + "<br>".join(problems)
    return f"<div style='font-family:monospace;line-height:1.6;font-size:15px;'><b>{status}</b><hr>" + "".join(strategy_html) + "</div>"

@app.route("/diag/e2e")
def diag_e2e():
    try:
        group = int(request.args.get("group", "-1"))
        # 🔧 FIX: request.get → request.args.get
        view = request.args.get("view", "json").lower()
        cumulative = request.args.get("cum", "0") == "1"
        symbols = request.args.get("symbols")
        out = diag_e2e_run(group=group, view=view, cumulative=cumulative, symbols=symbols)
        if isinstance(out, Response):
            return out
        if view == "html":
            return Response(out if isinstance(out, str) else str(out),
                            mimetype="text/html; charset=utf-8")
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/run")
def run():
    try:
        if os.path.exists(LOCK_PATH) or _is_training() or _is_group_active_file() or os.path.exists(GROUP_TRAIN_LOCK):
            return "⏸️ 학습/초기화/그룹학습 진행 중: 예측 시작 차단됨", 423
        print("[RUN] 전략별 예측 실행"); sys.stdout.flush()
        _pl_clear()
        _safe_open_gate("route_run")
        try:
            for strategy in ["단기","중기","장기"]:
                main(strategy, force=True)
        finally:
            _safe_close_gate("route_run")
        return "Recommendation started"
    except Exception as e:
        traceback.print_exc(); return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    try:
        if os.path.exists(LOCK_PATH):
            return "⏸️ 초기화 중: 학습 시작 차단됨", 423
        force = request.args.get("force", "0") == "1"
        _safe_close_gate("train_now_start")
        started = _start_train_loop_safe(force_restart=force, sleep_sec=0)
        if started: return "✅ 전체 그룹 학습 루프 시작됨 (백그라운드)"
        return "⏳ 이미 실행 중 (재가동 생략)" if not force else "⏳ 재가동 시도했으나 기존 루프 유지됨"
    except Exception as e:
        return f"학습 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        log_path = get_TRAIN_LOG_PATH()
        if not os.path.exists(log_path):
            return f"학습 로그 없음<br><small>경로: <code>{log_path}</code></small>"
        df = pd.read_csv(log_path, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty or df.shape[1] == 0:
            return f"학습 기록 없음<br><small>경로: <code>{log_path}</code></small>"
        html = df.tail(200).to_html(index=False, border=1, justify='center')
        return f"<b>📘 학습 로그 (최근 200행)</b><br><small>경로: <code>{log_path}</code></small><br><br>{html}"
    except Exception as e:
        return f"읽기 오류: {e}", 500

@app.route("/models")
def list_models():
    try:
        files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
        return "<pre>" + "\n".join(files) + "</pre>" if files else "models 폴더 비어 있음"
    except Exception as e:
        return f"오류: {e}", 500

@app.route("/check-log-full")
def check_log_full():
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        latest = df.sort_values(by="timestamp", ascending=False).head(100)
        return jsonify(latest.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG):
            return jsonify({"error": "prediction_log.csv 없음"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        if "timestamp" not in df:
            return jsonify([])
        df = df[df["timestamp"].notna()]
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-eval-log")
def check_eval_log():
    try:
        path = PREDICTION_LOG
        if not os.path.exists(path): return "예측 로그 없음"
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        if "status" not in df.columns: return "상태 컬럼 없음"
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        latest = df.sort_values(by="timestamp", ascending=False).head(100)
        def status_icon(s):
            return {"success":"✅","fail":"❌","v_success":"✅","v_fail":"❌","pending":"⏳","v_pending":"⏳"}.get(s,"❓")
        html = "<table border='1'><tr><th>시각</th><th>심볼</th><th>전략</th><th>모델</th><th>예측</th><th>수익률</th><th>상태</th><th>사유</th></tr>"
        for _, r in latest.iterrows():
            icon = status_icon(r.get("status",""))
            rv = r.get('return_value', None)
            if pd.isna(rv) or rv is None: rv = r.get('rate', 0)
            try: rv = float(rv)
            except Exception: rv = 0.0
            html += f"<tr><td>{r.get('timestamp','')}</td><td>{r.get('symbol','')}</td><td>{r.get('strategy','')}</td><td>{r.get('model','')}</td><td>{r.get('direction','')}</td><td>{rv:.4f}</td><td>{icon}</td><td>{r.get('reason','')}</td></tr>"
        html += "</table>"
        return html
    except Exception as e:
        return f"❌ 오류: {e}", 500

@app.route("/train-symbols", methods=["GET","POST"])
def train_symbols():
    try:
        if os.path.exists(LOCK_PATH):
            return f"⏸️ 초기화 중: 그룹/선택 학습 시작 차단됨", 423

        def _ensure_single_loop(force_flag: bool):
            if _is_training():
                if not force_flag:
                    return False, ("🚫 이미 메인 학습 루프 실행 중 (force=1 또는 force=true 로 강제 교체 가능)", 409)
                try:
                    _request_stop_safe(); _stop_train_loop_safe(timeout=45)
                except Exception:
                    pass
            return True, None

        if request.method == "GET":
            group_idx = int(request.args.get("group", -1))
            force = request.args.get("force", "0") == "1"
            if group_idx < 0 or group_idx >= len(SYMBOL_GROUPS):
                return f"❌ 잘못된 그룹 번호: {group_idx}", 400
            ok, resp = _ensure_single_loop(force)
            if not ok: return resp
            group_symbols = SYMBOL_GROUPS[group_idx]
            print(f"🚀 그룹 학습 요청됨 → 그룹 #{group_idx} | 심볼: {group_symbols}")
            # 그룹 시작: 게이트 닫기 + GROUP_ACTIVE 생성 + 그룹학습 락 생성
            _safe_close_gate("train_group_start")
            _set_group_active(True, group_idx=group_idx, symbols=group_symbols)
            try:
                with open(GROUP_TRAIN_LOCK, "w", encoding="utf-8") as f:
                    f.write(f"group={group_idx} started={datetime.datetime.utcnow().isoformat()}\n")
                print("[GROUP-LOCK] created"); sys.stdout.flush()
            except Exception as e:
                print(f"[GROUP-LOCK] create failed: {e}"); sys.stdout.flush()

            def _worker():
                try:
                    _train_models_safe(group_symbols)
                    if not group_all_complete():
                        print("[GROUP-AFTER] 미완료: group_all_complete()=False → 예측 생략"); return
                    if not ready_for_group_predict():
                        print("[GROUP-AFTER] 미완료: ready_for_group_predict()=False → 예측 생략"); return
                    _predict_after_training(group_symbols, source_note=f"group{group_idx}_after_train")
                    try:
                        mark_group_predicted(); print("[GROUP-AFTER] mark_group_predicted() 호출 완료")
                    except Exception as e:
                        print(f"[GROUP-AFTER] mark_group_predicted 예외: {e}")
                finally:
                    # 종료 시점에만 GROUP_ACTIVE 삭제 + 그룹학습 락 제거
                    if group_all_complete() and ready_for_group_predict():
                        _set_group_active(False)
                    try:
                        if os.path.exists(GROUP_TRAIN_LOCK):
                            os.remove(GROUP_TRAIN_LOCK)
                            print("[GROUP-LOCK] removed"); sys.stdout.flush()
                    except Exception as e:
                        print(f"[GROUP-LOCK] remove failed: {e}"); sys.stdout.flush()

            threading.Thread(target=_worker, daemon=True).start()
            return f"✅ 그룹 #{group_idx} 학습 시작됨 (완료 검증 통과 시 학습 직후 예측, 이후 mark_group_predicted)"
        else:
            body = request.get_json(silent=True) or {}
            symbols = body.get("symbols", [])
            force = bool(body.get("force", False))
            if not isinstance(symbols, list) or not symbols:
                return "❌ 유효하지 않은 symbols 리스트", 400
            ok, resp = _ensure_single_loop(force)
            if not ok: return resp
            # 선택 학습은 그룹 경계 아님 → GROUP_ACTIVE 비조작, 그룹 락 비사용
            _safe_close_gate("train_selected_start")
            def _worker():
                try:
                    _train_models_safe(symbols)
                    _predict_after_training(symbols, source_note="selected_after_train")
                finally:
                    pass
            threading.Thread(target=_worker, daemon=True).start()
            return f"✅ {len(symbols)}개 심볼 학습 시작됨 (학습 직후 예측 수행 — 그룹 마킹 없음)"
    except Exception as e:
        traceback.print_exc(); return f"❌ 오류: {e}", 500

@app.route("/meta-fix-now")
def meta_fix_now():
    try:
        maintenance_fix_meta.fix_all_meta_json()
        return "✅ meta.json 점검/복구 완료"
    except Exception as e:
        return f"⚠️ 실패: {e}", 500

@app.route("/reset-all", methods=["GET","POST"])
@app.route("/reset-all/<key>", methods=["GET","POST"])
def reset_all(key=None):
    req_key = key or request.args.get("key") or (request.json.get("key") if request.is_json else None)
    if req_key != "3572":
        print(f"[RESET] 인증 실패 from {request.remote_addr} path={request.path}"); sys.stdout.flush()
        return "❌ 인증 실패", 403
    ua = request.headers.get("User-Agent", "-")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    print(f"[RESET] 요청 수신 from {ip} UA={ua}"); sys.stdout.flush()
    _safe_close_gate("reset_enter")
    def _arm_reset_watchdog(seconds: int):
        seconds = max(30, int(seconds))
        def _kill():
            print(f"🛑 [WATCHDOG] reset watchdog fired after {seconds}s → os._exit(0)"); sys.stdout.flush()
            try: _release_global_lock()
            finally: os._exit(0)
        t = threading.Timer(seconds, _kill); t.daemon = True; t.start()
        return t
    def _do_reset_work():
        stop_timeout = int(os.getenv("RESET_STOP_TIMEOUT", "12"))
        max_wait     = int(os.getenv("RESET_MAX_WAIT_SEC", "120"))
        poll_sec     = max(1, int(os.getenv("RESET_POLL_SEC", "2")))
        watchdog_sec = int(os.getenv("RESET_WATCHDOG_SEC", str(stop_timeout + max_wait + 30)))
        qwipe_early  = os.getenv("RESET_QWIPE_EARLY", "1") == "1"
        _wd = _arm_reset_watchdog(watchdog_sec)
        try:
            from data.utils import _kline_cache, _feature_cache
            import importlib
            _acquire_global_lock()
            _stop_all_aux_schedulers()
            _pl_clear()
            PERSIST = PERSIST_DIR; LOGS = LOG_DIR; MODELS = MODEL_DIR
            def clear_csv(f, headers):
                os.makedirs(os.path.dirname(f), exist_ok=True)
                with open(f, "w", newline="", encoding="utf-8-sig") as wf:
                    wf.write(",".join(headers) + "\n")
            print("[RESET] 백그라운드 초기화 시작"); sys.stdout.flush()
            try:
                if hasattr(train, "request_stop"):
                    _request_stop_safe()
            except Exception: pass
            stopped = False
            try:
                print(f"[RESET] 학습 루프 정지 시도(timeout={stop_timeout}s)"); sys.stdout.flush()
                stopped = _stop_train_loop_safe(timeout=stop_timeout)
            except Exception as e:
                print(f"⚠️ [RESET] stop_train_loop 예외: {e}"); sys.stdout.flush()
            print(f"[RESET] stop_train_loop 결과: {stopped}"); sys.stdout.flush()
            if (not stopped) and qwipe_early:
                try:
                    print("[RESET] 빠른 정지 실패 → 조기 QWIPE 수행"); sys.stdout.flush()
                    _quarantine_wipe_persistent()
                    ensure_dirs()  # ✅ 조기 QWIPE 직후 즉시 복구
                except Exception as e:
                    print(f"⚠️ [RESET] 조기 QWIPE 실패: {e}"); sys.stdout.flush()
            if not stopped:
                t0 = time.time()
                print(f"[RESET] 정지 대기 시작… 최대 {max_wait}s (폴링 {poll_sec}s)"); sys.stdout.flush()
                while time.time() - t0 < max_wait:
                    try:
                        if not _is_training(): stopped = True; break
                    except Exception: pass
                    try:
                        if _stop_train_loop_safe(timeout=2): stopped = True; break
                    except Exception: pass
                    time.sleep(poll_sec)
                print(f"[RESET] 정지 대기 완료 → stopped={stopped}"); sys.stdout.flush()
            if not stopped:
                print("🛑 [RESET] 루프 미종료 → QWIPE 후 하드 종료"); sys.stdout.flush()
                try:
                    _quarantine_wipe_persistent()
                    ensure_dirs()
                except Exception as e: print(f"⚠️ [RESET] QWIPE 실패: {e}")
                try: _release_global_lock()
                finally:
                    try: _wd.cancel()
                    except Exception: pass
                    os._exit(0)
            try:
                done_path = os.path.join(PERSIST_DIR, "train_done.json")
                if os.path.exists(done_path): os.remove(done_path)
            except Exception: pass
            try:
                for d in [MODELS, LOGS, os.path.join(PERSIST_DIR, "ssl_models")]:
                    if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
                    os.makedirs(d, exist_ok=True)
                keep = {os.path.basename(LOCK_DIR)}
                for name in list(os.listdir(PERSIST_DIR)):
                    p = os.path.join(PERSIST_DIR, name)
                    if name in keep: continue
                    if os.path.isdir(p):
                        if name not in {"logs", "models", "ssl_models", "run"}: shutil.rmtree(p, ignore_errors=True)
                        if name == "run":
                            try:
                                for f in os.listdir(p):
                                    try: os.remove(os.path.join(p, f))
                                    except Exception: pass
                            except Exception: pass
                    else:
                        try: os.remove(p)
                        except Exception: pass
                suspect_prefixes = ("prediction_log","eval","message_log","train_log","wrong_predictions",
                                    "evaluation_audit","failure_count","diag","e2e","guan","관우")
                for root, dirs, files in os.walk(PERSIST_DIR, topdown=False):
                    for f in files:
                        low = f.lower()
                        if low.endswith((".csv",".db",".json",".txt")) or low.startswith(suspect_prefixes):
                            try: os.remove(os.path.join(root, f))
                            except Exception: pass
                    for d in dirs:
                        low = d.lower()
                        if low.startswith(suspect_prefixes) or ("관우" in d):
                            try: shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                            except Exception: pass
                # ✅ 풀와이프 이후에도 필수 경로 복구
                ensure_dirs()
            except Exception as e:
                print(f"⚠️ [RESET] 풀와이프 예외: {e}"); sys.stdout.flush()
            try: _kline_cache.clear()
            except Exception: pass
            try: _feature_cache.clear()
            except Exception: pass
            try:
                ensure_prediction_log_exists()
                ensure_train_log_exists()
                clear_csv(os.path.join(PERSIST_DIR, "wrong_predictions.csv"), PREDICTION_HEADERS)
                clear_csv(os.path.join(LOG_DIR, "evaluation_audit.csv"), ["timestamp","symbol","strategy","status","reason"])
                clear_csv(os.path.join(LOG_DIR, "message_log.csv"), ["timestamp","symbol","strategy","message"])
                clear_csv(os.path.join(LOG_DIR, "failure_count.csv"), ["symbol","strategy","failures"])
            except Exception as e:
                print(f"⚠️ [RESET] 로그 재생성 예외: {e}"); sys.stdout.flush()
            try:
                import diag_e2e as _diag_mod, importlib as _imp
                _imp.reload(_diag_mod)
            except Exception: pass
            try:
                maintenance_fix_meta.fix_all_meta_json()
            except Exception as e:
                print(f"[RESET] meta 보정 실패: {e}")
            print("🔚 [RESET] 정리 완료 → 프로세스 종료(os._exit)"); sys.stdout.flush()
            _release_global_lock()
            try: _wd.cancel()
            except Exception: pass
            os._exit(0)
        except Exception as e:
            print(f"❌ [RESET] 백그라운드 초기화 예외: {e}"); sys.stdout.flush()
        finally:
            _release_global_lock()
    threading.Thread(target=_do_reset_work, daemon=True).start()
    return Response(
        "✅ 초기화 요청 접수됨. 백그라운드에서 정지→정리 후 서버 프로세스를 재시작합니다.\n"
        "로그에서 [RESET]/[SCHED]/[LOCK]/[QWIPE] 태그를 확인하세요.",
        mimetype="text/plain; charset=utf-8"
    )

@app.route("/force-fix-prediction_log")
@app.route("/force-fix-prediction-log")
def force_fix_prediction_log():
    try:
        if os.path.exists(PREDICTION_LOG):
            os.remove(PREDICTION_LOG)
        ensure_prediction_log_exists()
        print("[FORCE-FIX] prediction_log.csv 재생성 완료"); sys.stdout.flush()
        return "✅ prediction_log.csv 강제 초기화 완료"
    except Exception as e:
        return f"⚠️ 오류: {e}", 500

# 즉시 전체 예측: 학습 일시정지 → 예측 → 학습 재개
@app.route("/predict-now", methods=["POST","GET"])
def predict_now():
    try:
        was_running = _is_training()
        if was_running:
            print("[PREDICT-NOW] training detected → stopping..."); sys.stdout.flush()
            try: _request_stop_safe()
            except Exception: pass
            stopped = _stop_train_loop_safe(timeout=45)
            if not stopped:
                return "❌ 학습 정지 실패로 예측 취소됨", 423

        if os.path.exists(LOCK_PATH) or _is_group_active_file() or os.path.exists(GROUP_TRAIN_LOCK):
            return "⏸️ 초기화/그룹학습 중: 예측 시작 차단됨", 423
        _pl_clear()
        _safe_open_gate("predict_now")

        total, done, skipped = 0, 0, 0
        try:
            for sym in SYMBOLS:
                for strat in ["단기","중기","장기"]:
                    total += 1
                    if not _has_model_for(sym, strat):
                        skipped += 1
                        print(f"[PREDICT-NOW] skip {sym}-{strat}: model missing"); sys.stdout.flush()
                        continue
                    try:
                        result = predict(sym, strat, source="predict_now", model_type=None)
                        if not isinstance(result, dict):
                            try:
                                from predict import failed_result
                                failed_result(sym, strat, reason="invalid_return", source="predict_now")
                            except Exception: pass
                        done += 1
                        print(f"[PREDICT-NOW] ok {sym}-{strat}"); sys.stdout.flush()
                    except Exception as e:
                        print(f"[PREDICT-NOW] fail {sym}-{strat}: {e}"); sys.stdout.flush()
                        try:
                            from predict import failed_result
                            failed_result(sym, strat, reason=str(e), source="predict_now")
                        except Exception: pass
        finally:
            _safe_close_gate("predict_now_end")

        resumed = False
        if was_running:
            resumed = _start_train_loop_safe(force_restart=False, sleep_sec=0)
            print(f"[PREDICT-NOW] training resumed={resumed}"); sys.stdout.flush()

        return f"✅ 예측 완료 | 총:{total} 성공:{done} 스킵:{skipped} | 학습정지:{'예' if was_running else '아니오'} → 재개:{'예' if resumed else '아니오'}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ 오류: {e}", 500


if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
    except ValueError:
        raise RuntimeError("❌ Render 환경변수 PORT가 없습니다. Render 서비스 타입 확인 필요")
    _init_background_once()
    print(f"✅ Flask 서버 실행 시작 (PORT={port})")
    app.run(host="0.0.0.0", port=port)
