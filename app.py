# app.py — FINAL v2.2f
# (reset-all 강제 풀초기화 버전, kline warmup fix)
# dirs auto-heal, PERSIST_DIR/PERSISTENT_DIR env, lock-dir PermissionError fallback
# train→predict→next-group 파이프라인, 부팅시 필수 경로/빈 로그 보장,
# 예측락 stale GC, 그룹학습 락/게이트

import sitecustomize  # 경로 자동변환 강제 로드

from flask import Flask, jsonify, request, Response
from recommend import main
import train
import os
import threading
import datetime
import pytz
import traceback
import sys
import shutil
import re
import time
import pandas as pd
from logger import get_train_log_cards, TRAIN_LOG


from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_trigger import run as trigger_run
from predict_trigger import _is_group_complete_for_all_strategies, _get_current_group_symbols
from data.utils import SYMBOLS, get_kline_by_strategy, CacheManager
from data.utils import (
    ready_for_group_predict,
    mark_group_predicted,
    group_all_complete,
    get_current_group_symbols,
    SYMBOL_GROUPS,
)
from visualization import generate_visual_report, generate_visuals_for_strategy
from wrong_data_loader import load_training_prediction_data
from predict import evaluate_predictions
from train import train_symbol_group_loop  # compatibility
import maintenance_fix_meta
from logger import (
    ensure_prediction_log_exists,
    ensure_train_log_exists,
    PREDICTION_HEADERS,
    TRAIN_HEADERS,
)
from logger import log_audit_prediction as log_audit
from config import get_TRAIN_LOG_PATH


# === 공통 경로/디렉토리 ===
# NOTE: Render에서는 /persistent 쓰면 Permission denied가 뜨므로 기본값을 /tmp/persistent 로 둔다.
# 로컬/자체 서버에서 예전처럼 /persistent 쓰고 싶으면
# PERSIST_DIR=/persistent 또는 PERSISTENT_DIR=/persistent 로 환경변수만 주면 된다.

PERSIST_DIR = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/tmp/persistent"
)
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
RUN_DIR = os.path.join(PERSIST_DIR, "run")

os.makedirs(PERSIST_DIR, exist_ok=True)

# 예전 하드코딩 로그 파일들 미리 생성
for name in ("wrong_predictions.csv", "prediction_log.csv", "train_log.csv"):
    path = os.path.join(PERSIST_DIR, name)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("")


# --- ✨ 필수 폴더 자동 생성 유틸 (부팅/리셋/조기 QWIPE 후 재사용) ---

NEEDED_DIRS = [
    f"{PERSIST_DIR}/importances",
    f"{PERSIST_DIR}/guanwu/incoming",
    LOG_DIR,
    MODEL_DIR,
    RUN_DIR,
    "/tmp/importances",
    # 👇 옛날 코드가 하드코딩으로 쓰는 경로도 같이 만들어서 로그 경고 제거
    "/persistent/importances",
]


def ensure_dirs():
    for p in NEEDED_DIRS:
        os.makedirs(p, exist_ok=True)


# 모듈 로드 시 1회 보장
ensure_dirs()


# === integrity guard optional ===
try:
    from integrity_guard import run as _integrity_check

    _integrity_check()
except Exception as e:
    print(f"[WARN] integrity_guard skipped: {e}")


from diag_e2e import run as diag_e2e_run


# === cleanup modules ===
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
        safe_cleanup = type(
            "sc",
            (),
            {
                "get_directory_size_gb": staticmethod(lambda p: 0),
                "HARD_CAP_GB": 9.6,
                "run_emergency_purge": staticmethod(lambda: None),
                "cleanup_logs_and_models": staticmethod(lambda: None),
            },
        )
        _cleanup_mod = safe_cleanup


# === predict-lock stale GC ===
try:
    import predict_lock as _pl

    _pl_clear = getattr(_pl, "clear_stale_predict_lock", lambda: None)
except Exception:
    _pl = None
    _pl_clear = lambda: None


# === predict gate ===
try:
    from predict import open_predict_gate, close_predict_gate, predict
except Exception:
    def open_predict_gate(*args, **kwargs):
        return None

    def close_predict_gate(*args, **kwargs):
        return None

    def predict(*args, **kwargs):
        raise RuntimeError("predict 불가")


def _safe_open_gate(note: str = ""):
    try:
        open_predict_gate(note=note)
        print(f"[gate] open ({note})")
        sys.stdout.flush()
    except Exception as e:
        print(f"[gate] open err: {e}")
        sys.stdout.flush()


def _safe_close_gate(note: str = ""):
    try:
        close_predict_gate(note=note)
        print(f"[gate] close ({note})")
        sys.stdout.flush()
    except Exception as e:
        print(f"[gate] close err: {e}")
        sys.stdout.flush()


# [ADD] 그룹잠금 전용 파일
GROUP_TRAIN_LOCK = os.path.join(RUN_DIR, "group_training.lock")


# === locks — PermissionError 대비 경로 고정 ===
_lock_dir_candidate = getattr(safe_cleanup, "LOCK_DIR", os.path.join(PERSIST_DIR, "locks"))
try:
    os.makedirs(_lock_dir_candidate, exist_ok=True)
    LOCK_DIR = _lock_dir_candidate
except PermissionError:
    # Render에서 /persistent 쪽이 막혀 있을 때 여기로 폴백
    LOCK_DIR = os.path.join(PERSIST_DIR, "locks_local")
    os.makedirs(LOCK_DIR, exist_ok=True)

# safe_cleanup에 LOCK_PATH가 있으면 그걸 쓰고, 아니면 우리가 방금 확정한 LOCK_DIR을 쓴다.
LOCK_PATH = getattr(safe_cleanup, "LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))


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
            print(f"[GROUP_ACTIVE] set group={group_idx} symbols={symbols}")
            sys.stdout.flush()
        else:
            if os.path.exists(GROUP_ACTIVE_PATH):
                os.remove(GROUP_ACTIVE_PATH)
            print("[GROUP_ACTIVE] cleared")
            sys.stdout.flush()
    except Exception as e:
        print(f"[GROUP_ACTIVE] set/clear err: {e}")
        sys.stdout.flush()


def _is_group_active_file() -> bool:
    try:
        return os.path.exists(GROUP_ACTIVE_PATH)
    except Exception:
        return False


# === BOOT: orphan 전역락 제거 + 예측락 stale GC + 그룹학습락 제거 ===
if os.path.exists(LOCK_PATH):
    try:
        os.remove(LOCK_PATH)
        print("[BOOT] orphan lock removed")
        sys.stdout.flush()
    except Exception as e:
        print(f"[BOOT] lock remove failed: {e}")
        sys.stdout.flush()

try:
    _pl_clear()
    print("[BOOT] predict lock stale-GC done")
except Exception as e:
    print(f"[BOOT] predict lock GC failed: {e}")

try:
    if os.path.exists(GROUP_TRAIN_LOCK):
        os.remove(GROUP_TRAIN_LOCK)
        print("[BOOT] stale GROUP_TRAIN_LOCK removed")
        sys.stdout.flush()
except Exception as e:
    print(f"[BOOT] GROUP_TRAIN_LOCK cleanup failed: {e}")
    sys.stdout.flush()


def _acquire_global_lock():
    try:
        os.makedirs(LOCK_DIR, exist_ok=True)
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(f"locked at {datetime.datetime.now().isoformat()}\n")
        print(f"[LOCK] created: {LOCK_PATH}")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"[LOCK] create failed: {e}")
        sys.stdout.flush()
        return False


def _release_global_lock():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
        print(f"[LOCK] removed: {LOCK_PATH}")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"[LOCK] remove failed: {e}")
        sys.stdout.flush()
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
    """
    기존 문제:
    - train 루프가 내부에서 blocking 상태일 때 멈추지 않음
    - timeout 지나도 False만 반환 → reset-all 무한대기
    - 결국 초기화 전체 멈춤

    수정 내용:
    - request_stop() → stop_train_loop() 순으로 강제 시도
    - timeout 지나면 마지막으로 thread 강제 kill 시도
    """

    fn = getattr(train, "stop_train_loop", None)
    req = getattr(train, "request_stop", None)

    # 1) 일단 중단 요청
    try:
        if callable(req):
            req()
    except Exception:
        pass

    # 2) 정상 종료 대기
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            if not _is_training():
                return True
        except Exception:
            pass
        time.sleep(0.5)

    # 3) 그래도 안멈추면 강제 중단
    try:
        if callable(fn):
            fn(timeout=2)
    except Exception:
        pass

    # 4) 마지막 체크
    try:
        return not _is_training()
    except Exception:
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
    fn = getattr(train, "await_models_visible", None)
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
                        if f.startswith(f"{s}") and f.endswith(exts):
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


def has_model_for(symbol, strategy):
    fn = getattr(train, "has_model_for", None)
    if callable(fn):
        try:
            return bool(fn(symbol, strategy))
        except Exception:
            pass

    try:
        exts = (".pt", ".ptz", ".safetensors")
        pref = f"{symbol}{strategy}"
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


# ✅ 학습 직전에 거래소에서 최신 캔들 무조건 땡겨오게 하는 헬퍼 (캐시 먼저 삭제 후 정상 호출)
def _warmup_latest_klines(symbols):
    print("[APP] 최신 캔들 재수집 시작")
    for sym in symbols:
        for strat in ["단기", "중기", "장기"]:
            try:
                # 캐시키 패턴과 맞춰서 먼저 삭제
                try:
                    cache_key = f"{sym.upper()}-{strat}-slack0"
                    CacheManager.delete(cache_key)
                except Exception:
                    pass
                # 실제 새 데이터 가져오기
                get_kline_by_strategy(sym, strat, end_slack_min=0)
            except Exception as e:
                print(f"[APP] {sym}-{strat} 수집 실패: {e}")
    print("[APP] 최신 캔들 재수집 완료")
    sys.stdout.flush()


# === quarantine wipe helper ===
def _quarantine_wipe_persistent():
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trash_dir = os.path.join(PERSIST_DIR, f"trash{ts}")
    os.makedirs(trash_dir, exist_ok=True)
    keep_names = {os.path.basename(LOCK_DIR)}
    moved = []
    for name in list(os.listdir(PERSIST_DIR)):
        if name in keep_names:
            continue
        src = os.path.join(PERSIST_DIR, name)
        dst = os.path.join(trash_dir, name)
        try:
            shutil.move(src, dst)
            moved.append(name)
        except Exception as e:
            print(f"⚠️ [QWIPE] move 실패: {src} -> {dst} ({e})")
    # ✅ 핵심: 바로 재생성(학습/특징중요도 저장 중에도 디렉터리 존재 보장)
    ensure_dirs()
    print(f"🧨 [QWIPE] moved_to_trash={moved} trash_dir={trash_dir}")
    sys.stdout.flush()
    return trash_dir


def _async_emergency_purge():
    try:
        try:
            for name in list(os.listdir(PERSIST_DIR)):
                if name.startswith("trash"):
                    path = os.path.join(PERSIST_DIR, name)
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"[BOOT-CLEANUP] trashed removed: {name}")
        except Exception as e:
            print(f"⚠️ [BOOT-CLEANUP] trash 제거 실패: {e}")

        used_gb = safe_cleanup.get_directory_size_gb(PERSIST_DIR)
        hard_cap = getattr(safe_cleanup, "HARD_CAP_GB", 9.6)
        print(f"[BOOT-CLEANUP] used={used_gb:.2f}GB hard_cap={hard_cap:.2f}GB")
        sys.stdout.flush()
        if used_gb >= hard_cap:
            print("[EMERGENCY] pre-DB purge 시작 (하드캡 초과)")
            sys.stdout.flush()
            safe_cleanup.run_emergency_purge()
            print("[EMERGENCY] pre-DB purge 완료")
            sys.stdout.flush()
        else:
            if os.getenv("CLEANUP_ON_BOOT", "0") == "1":
                print("[BOOT-CLEANUP] CLEANUP_ON_BOOT=1 → 온건 정리 실행")
                sys.stdout.flush()
                safe_cleanup.cleanup_logs_and_models()
                print("[BOOT-CLEANUP] 완료")
                sys.stdout.flush()
            else:
                print("[BOOT-CLEANUP] 비활성화(CLEANUP_ON_BOOT=0)")
                sys.stdout.flush()
    except Exception as e:
        print(f"[경고] pre-DB purge/cleanup 결정 실패: {e}")
        sys.stdout.flush()


threading.Thread(target=_async_emergency_purge, daemon=True).start()

PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
LOG_FILE = get_TRAIN_LOG_PATH()
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_LOG = os.path.join(LOG_DIR, "failure_count.csv")


# ensure logs
try:
    ensure_train_log_exists()
except Exception:
    pass

ensure_prediction_log_exists()

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# 그룹 완주 후에만 예측 허용
REQUIRE_GROUP_COMPLETE = int(os.getenv("REQUIRE_GROUP_COMPLETE", "1"))


# === prediction_log 전용 필터 ===
def _filter_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    prediction_log.csv 안에 섞여 있는
    - 학습용 로그(model='trainer')
    - source 가 train* 로 시작하는 행
    을 화면/리포트에서 완전히 제거한다.
    """
    try:
        if df is None or df.empty:
            return df
        mask = pd.Series(True, index=df.index)

        if "model" in df.columns:
            mask &= df["model"].astype(str) != "trainer"

        if "source" in df.columns:
            s = df["source"].astype(str).str.lower()
            train_like = s.str.startswith("train")
            train_extra = s.isin(
                [
                    "trainer",
                    "train_return_distribution",
                    "train_dist",
                    "train_eval",
                    "train_bins",
                ]
            )
            mask &= ~(train_like | train_extra)

        # 전략 컬럼이 있으면 단기/중기/장기 만 남기기
        if "strategy" in df.columns:
            strat_s = df["strategy"].astype(str)
            mask &= strat_s.isin(["단기", "중기", "장기"])

        return df[mask]
    except Exception:
        return df


# === scheduler ===
_sched = None


def start_scheduler():
    global _sched
    if _sched is not None:
        print("⚠️ 스케줄러 이미 실행 중, 재시작 생략")
        sys.stdout.flush()
        return

    if os.path.exists(LOCK_PATH):
        print("⏸️ 리셋 락 감지 → 스케줄러 시작 지연")
        sys.stdout.flush()

        def _deferred():
            backoff = 1.0
            while os.path.exists(LOCK_PATH):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
            try:
                start_scheduler()
                print("▶️ 지연 후 스케줄러 시작")
                sys.stdout.flush()
            except Exception as e:
                print(f"❌ 지연 시작 실패: {e}")

        threading.Thread(target=_deferred, daemon=True).start()
        return

    try:
        _pl_clear()
        print("[SCHED] predict lock stale-GC pre-start")
    except Exception as e:
        print(f"[SCHED] predict lock GC failed pre-start: {e}")

    print(">>> 스케줄러 시작")
    sys.stdout.flush()
    sched = BackgroundScheduler(
        timezone=pytz.timezone("Asia/Seoul"),
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 90},
    )

    def 평가작업(strategy):
        def wrapped():
            try:
                if (
                    _is_training()
                    or os.path.exists(LOCK_PATH)
                    or _is_group_active_file()
                    or os.path.exists(GROUP_TRAIN_LOCK)
                ):
                    print(f"[EVAL] skip: training/lock/group-active (strategy={strategy})")
                    sys.stdout.flush()
                    return
                ts = now_kst().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[EVAL][{ts}] 전략={strategy} 시작")
                sys.stdout.flush()
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
            if (
                _is_training()
                or os.path.exists(LOCK_PATH)
                or _is_group_active_file()
                or os.path.exists(GROUP_TRAIN_LOCK)
            ):
                print("[PREDICT] skip: training/lock/group-active")
                sys.stdout.flush()
                return
            _pl_clear()
            print("[PREDICT] trigger_run start")
            sys.stdout.flush()
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
            print("[PREDICT] trigger_run done")
            sys.stdout.flush()
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
    print("✅ 스케줄러 시작 완료")
    sys.stdout.flush()


def _pause_and_clear_scheduler():
    global _sched
    try:
        if _sched is not None:
            print("[SCHED] pause + remove_all_jobs")
            sys.stdout.flush()
            try:
                _sched.pause()
            except Exception:
                pass
            try:
                _sched.remove_all_jobs()
            except Exception:
                pass
            try:
                _sched.shutdown(wait=False)
            except Exception:
                pass
            _sched = None
    except Exception as e:
        print(f"[SCHED] 정지 실패: {e}")
        sys.stdout.flush()


def _stop_all_aux_schedulers():
    try:
        _pause_and_clear_scheduler()
    except Exception:
        pass
    try:
        if hasattr(_cleanup_mod, "stop_cleanup_scheduler"):
            try:
                _cleanup_mod.stop_cleanup_scheduler()
                print("🧹 [SCHED] cleanup 스케줄러 stop 호출")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ cleanup stop 실패: {e}")
                sys.stdout.flush()
        for name in dir(_cleanup_mod):
            obj = getattr(_cleanup_mod, name, None)
            if isinstance(obj, BackgroundScheduler):
                try:
                    obj.shutdown(wait=False)
                    print(f"🧹 [SCHED] cleanup.{name} shutdown")
                    sys.stdout.flush()
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ cleanup 스케줄러 탐지 실패: {e}")
        sys.stdout.flush()


# === Flask app ===

app = Flask(__name__)
print(">>> Flask 앱 생성 완료")
sys.stdout.flush()

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
                print("⏸️ 락 감지 → 백그라운드 초기화 지연")
                sys.stdout.flush()
                return

            from failure_db import ensure_failure_db

            print(">>> 서버 실행 준비")
            ensure_failure_db()
            print("✅ failure_patterns DB 초기화 완료")

            # 필수 경로 및 빈 로그 보장(폴더는 이미 생성됨)
            try:
                ensure_train_log_exists()
            except Exception:
                pass
            try:
                ensure_prediction_log_exists()
            except Exception:
                pass

            _pl_clear()
            print("[pipeline] serialized: train -> predict -> next-group")
            sys.stdout.flush()

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

            # 🔔 여기서 항상 텔레그램 알림 보냄
            try:
                send_message("[시작] YOPO 서버 실행됨")
                print("✅ Telegram 알림 발송 완료")
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


# === 학습 직후 예측: 그룹 완주 가드 ===

def _predict_after_training(symbols, source_note):
    """
    학습이 끝난 뒤 바로 예측을 붙일 때 쓰는 헬퍼.
    → 하지만 학습 마커/락이 아직 살아 있으면 바로 예측하지 말고 스킵한다.
    """
    if not symbols:
        return

    # 학습 중 여부를 PERSIST_DIR 기준으로 다시 한 번 강하게 확인
    busy_flags = [
        GROUP_ACTIVE_PATH,                              # PERSIST_DIR/GROUP_ACTIVE
        os.path.join(RUN_DIR, "group_training.lock"),   # PERSIST_DIR/run/group_training.lock
        os.path.join(RUN_DIR, "group_predict.active"),  # 혹시 있을 옛날 마커
    ]
    for p in busy_flags:
        if os.path.exists(p):
            print(f"[APP-PRED] 🚫 학습/그룹 락 감지({p}) → 예측 전체 스킵 ({source_note})")
            for sym in sorted(set(symbols)):
                try:
                    log_audit(sym, "ALL", "학습후예측스킵", f"학습락존재:{os.path.basename(p)}")
                except Exception:
                    pass
            return

    # 그룹 완주 옵션이 켜져 있으면 여기서도 검사
    try:
        if REQUIRE_GROUP_COMPLETE == 1:
            group_syms = _get_current_group_symbols()
            if isinstance(group_syms, (list, tuple)) and len(group_syms) > 0:
                if not _is_group_complete_for_all_strategies(list(group_syms)):
                    print(f"[APP-PRED] 🚫 그룹 미완료 → 예측 스킵 ({source_note})")
                    for sym in sorted(set(symbols)):
                        try:
                            log_audit(sym, "ALL", "학습후예측스킵", "그룹미완료(REQUIRE_GROUP_COMPLETE=1)")
                        except Exception:
                            pass
                    return
    except Exception as e:
        print(f"[APP-PRED] 그룹완료검증 실패: {e}")

    # 모델 파일이 실제로 보일 때까지 잠깐 기다림
    try:
        await_sec = int(os.getenv("PREDICT_MODEL_AWAIT_SEC", "60"))
    except Exception:
        await_sec = 60
    visible_syms = _await_models_visible(symbols, timeout_sec=await_sec)
    if not visible_syms:
        print(f"[APP-PRED] 모델 가시화 실패 → 예측 생략 candidates={sorted(set(symbols))}")
        return

    # 혹시 남아 있는 전역락 제거
    if os.path.exists(LOCK_PATH):
        try:
            os.remove(LOCK_PATH)
            print("[APP-PRED] cleared stale lock before predict")
            sys.stdout.flush()
        except Exception as e:
            print(f"[APP-PRED] lock remove failed: {e}")
            sys.stdout.flush()

    _pl_clear()
    _safe_open_gate(source_note)
    try:
        for sym in sorted(set(visible_syms)):
            for strat in ["단기", "중기", "장기"]:
                try:
                    if not has_model_for(sym, strat):
                        print(f"[APP-PRED] skip {sym}-{strat}: model missing")
                        continue
                    print(f"[APP-PRED] predict {sym}-{strat}")
                    try:
                        result = predict(sym, strat, source=source_note, model_type=None)
                        if not isinstance(result, dict):
                            print(f"[APP-PRED] ⚠️ invalid return")
                            sys.stdout.flush()
                            try:
                                from predict import failed_result

                                failed_result(sym, strat, reason="invalid_return", source="app_predict")
                            except Exception:
                                pass
                        else:
                            print(f"[APP-PRED] ✅ {sym}-{strat} ok: {result.get('reason', 'ok')}")
                            sys.stdout.flush()
                    except Exception as e:
                        print(f"[APP-PRED] ❌ 예측 오류: {e}")
                        sys.stdout.flush()
                        try:
                            from predict import failed_result

                            failed_result(sym, strat, reason=str(e), source="app_predict")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[APP-PRED] {sym}-{strat} 실패: {e}")
                    sys.stdout.flush()
    finally:
        _safe_close_gate(source_note + "_end")


# === routes ===

@app.route("/")
def index():
    return "Yopo server is running"


@app.route("/ping")
def ping():
    return "pong"


@app.route("/admin/clear-predict-lock", methods=["POST", "GET"])
def clear_predict_lock_admin():
    try:
        _pl_clear()
        return "✅ predict lock stale-GC executed"
    except Exception as e:
        return f"⚠️ fail: {e}", 500


@app.route("/yopo-health")
def yopo_health():
    logs, strategy_html, problems = {}, [], []
    file_map = {
        "pred": PREDICTION_LOG,
        "train": LOG_FILE,
        "audit": AUDIT_LOG,
        "msg": MESSAGE_LOG,
    }
    for name, path in file_map.items():
        try:
            logs[name] = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
            if "timestamp" in logs[name].columns:
                logs[name] = logs[name][logs[name]["timestamp"].notna()]
            # 🔍 prediction_log는 여기서도 학습용 행 제외
            if name == "pred":
                logs[name] = _filter_prediction_rows(logs[name])
        except Exception:
            logs[name] = pd.DataFrame()

    try:
        model_files = [
            f for f in os.listdir(MODEL_DIR)
            if f.endswith((".pt", ".ptz", ".safetensors"))
        ]
    except Exception:
        model_files = []

    model_info = {}
    for f in model_files:
        m = re.match(
            r"(.+?)(단기|중기|장기)(lstm|cnn_lstm|transformer)(?:.*)?\.(pt|ptz|safetensors)$",
            f,
        )
        if m:
            symbol, strat, mtype, ext = m.groups()
            model_info.setdefault(strat, {}).setdefault(symbol, set()).add(mtype)

    for strat in ["단기", "중기", "장기"]:
        try:
            pred = logs.get("pred", pd.DataFrame())
            train_log_df = logs.get("train", pd.DataFrame())
            audit = logs.get("audit", pd.DataFrame())

            pred = pred.query(f"strategy == '{strat}'") if not pred.empty else pd.DataFrame()
            train_log_q = (
                train_log_df.query(f"strategy == '{strat}'")
                if not train_log_df.empty
                else pd.DataFrame()
            )
            audit = audit.query(f"strategy == '{strat}'") if not audit.empty else pd.DataFrame()

            if not pred.empty and "status" in pred.columns:
                pred["volatility"] = pred["status"].astype(str).str.startswith("v")
            else:
                pred["volatility"] = False

            try:
                pred["return"] = pd.to_numeric(
                    pred.get("return_value", pred.get("rate", pd.Series(dtype=float))),
                    errors="coerce",
                ).fillna(0.0)
            except Exception:
                pred["return"] = 0.0

            nvol = pred[~pred["volatility"]] if not pred.empty else pd.DataFrame()
            vol = pred[pred["volatility"]] if not pred.empty else pd.DataFrame()

            def stat(df, s):
                try:
                    return int(
                        ((not df.empty) and ("status" in df.columns))
                        and (df["status"] == s).sum()
                    ) or 0
                except Exception:
                    return 0

            sn, fn, pn_, fnl = map(lambda s: stat(nvol, s), ["success", "fail", "pending", "failed"])
            sv, fv, pv, fvl = map(lambda s: stat(vol, s), ["v_success", "v_fail", "pending", "failed"])

            def perf(df, kind="일반"):
                try:
                    s = stat(df, "v_success" if kind == "변동성" else "success")
                    f = stat(df, "v_fail" if kind == "변동성" else "fail")
                    t = s + f
                    avg = (
                        float(df["return"].mean())
                        if ("return" in df) and (df.shape[0] > 0)
                        else 0.0
                    )
                    return {
                        "succ": s,
                        "fail": f,
                        "succ_rate": (s / t * 100) if t else 0,
                        "fail_rate": (f / t * 100) if t else 0,
                        "r_avg": avg,
                        "total": t,
                    }
                except Exception:
                    return {
                        "succ": 0,
                        "fail": 0,
                        "succ_rate": 0,
                        "fail_rate": 0,
                        "r_avg": 0,
                        "total": 0,
                    }

            pn = perf(nvol, "일반")
            pv_stats = perf(vol, "변동성")

            strat_models = model_info.get(strat, {})
            types = {"lstm": 0, "cnn_lstm": 0, "transformer": 0}
            for mtypes in strat_models.values():
                for t in mtypes:
                    types[t] += 1
            trained_syms = [
                s for s, t in strat_models.items()
                if {"lstm", "cnn_lstm", "transformer"}.issubset(t)
            ]
            try:
                untrained = sorted(set(SYMBOLS) - set(trained_syms))
            except Exception:
                untrained = []

            if sum(types.values()) == 0:
                problems.append(f"{strat}: 모델 없음")
            if sn + fn + pn_ + fnl + sv + fv + pv + fvl == 0:
                problems.append(f"{strat}: 예측 없음")
            if pn["total"] == 0:
                problems.append(f"{strat}: 평가 미작동")
            if pn["fail_rate"] > 50:
                problems.append(f"{strat}: 일반 실패율 {pn['fail_rate']:.1f}%")
            if pv_stats["fail_rate"] > 50:
                problems.append(f"{strat}: 변동성 실패율 {pv_stats['fail_rate']:.1f}%")

            table = "<i style='color:gray'>최근 예측 없음 또는 컬럼 부족</i>"
            required_cols = {"timestamp", "symbol", "strategy", "direction", "return", "status"}
            if (pred.shape[0] > 0) and required_cols.issubset(set(pred.columns)):
                recent10 = pred.sort_values("timestamp").tail(10).copy()
                rows = []
                for _, r in recent10.iterrows():
                    rtn = r.get("return", 0.0) or r.get("rate", 0.0)
                    try:
                        rtn_pct = f"{float(rtn) * 100:.2f}%"
                    except Exception:
                        rtn_pct = "0.00%"
                    s = str(r.get("status", ""))
                    status_icon = (
                        "✅"
                        if s in ["success", "v_success"]
                        else "❌"
                        if s in ["fail", "v_fail"]
                        else "⏳"
                        if s in ["pending", "v_pending"]
                        else "🛑"
                    )
                    rows.append(
                        f"<tr><td>{r.get('timestamp','')}</td>"
                        f"<td>{r.get('symbol','')}</td>"
                        f"<td>{r.get('strategy','')}</td>"
                        f"<td>{r.get('direction','')}</td>"
                        f"<td>{rtn_pct}</td><td>{status_icon}</td></tr>"
                    )
                table = (
                    "<table border='1' style='margin-top:4px'>"
                    "<tr><th>시각</th><th>심볼</th><th>전략</th>"
                    "<th>방향</th><th>수익률</th><th>상태</th></tr>"
                    + "".join(rows)
                    + "</table>"
                )

            last_train = (
                train_log_df["timestamp"].iloc[-1]
                if (not train_log_df.empty and "timestamp" in train_log_df)
                else "없음"
            )
            last_pred = (
                pred["timestamp"].iloc[-1]
                if (not pred.empty and "timestamp" in pred)
                else "없음"
            )
            last_audit = (
                audit["timestamp"].iloc[-1]
                if (not audit.empty and "timestamp" in audit)
                else "없음"
            )

            info_html = f"""
<div style='border:1px solid #aaa;margin:16px 0;padding:10px;
            font-family:monospace;background:#f8f8f8;'>
<b style='font-size:16px;'>📌 전략: {strat}</b><br>
모델 수: {sum(types.values())} (lstm={types['lstm']}, cnn={types['cnn_lstm']}, trans={types['transformer']})<br>
심볼 수: {len(SYMBOLS)} | 완전학습: {len(trained_syms)} | 미완성: {len(untrained)}<br>
최근 학습: {last_train}<br>
최근 예측: {last_pred}<br>
최근 평가: {last_audit}<br>
예측 (일반): {sn + fn + pn_ + fnl}건 (✅{sn} ❌{fn} ⏳{pn_} 🛑{fnl})<br>
예측 (변동성): {sv + fv + pv + fvl}건 (✅{sv} ❌{fv} ⏳{pv} 🛑{fvl})<br>
<b style='color:#000088'>🎯 일반 예측</b>: {pn['total']}건 |
{pn['succ_rate']:.1f}% / {pn['fail_rate']:.1f}% / {pn['r_avg']:.2f}%<br>
<b style='color:#880000'>🌪️ 변동성 예측</b>: {pv_stats['total']}건 |
{pv_stats['succ_rate']:.1f}% / {pv_stats['fail_rate']:.1f}% / {pv_stats['r_avg']:.2f}%<br>
<b>📋 최근 예측 10건</b><br>{table}
</div>"""

            try:
                visual = generate_visuals_for_strategy(strat)
            except Exception as e:
                visual = f"<div style='color:red'>[시각화 실패: {e}]</div>"

            strategy_html.append(
                f"<div style='margin-bottom:30px'>{info_html}"
                f"<div style='margin:20px 0'>{visual}</div><hr></div>"
            )
        except Exception as e:
            strategy_html.append(
                f"<div style='color:red;'>❌ {strat} 실패: {type(e).__name__} → {e}</div>"
            )

    status = (
        "🟢 전체 전략 정상 작동 중"
        if not problems
        else "🔴 종합진단 요약:<br>" + "<br>".join(problems)
    )
    return (
        "<div style='font-family:monospace;line-height:1.6;font-size:15px;'>"
        f"<b>{status}</b><hr>"
        + "".join(strategy_html)
        + "</div>"
    )


# =========================
# 예측 + 평가 통합 대시보드 헬퍼
# =========================
def _render_prediction_eval_dashboard_simple():
    """
    prediction_log.csv 기준
    - 최근 100건 예측을 가져와서
    - 예측이 잘 찍히는지
    - 평가가 잘 되어 성공/실패로 끝났는지
    - 성공/실패/보류 사유가 무엇인지
    - 하이브리드/유사도 기반 선택 사유( note JSON )까지 같이 보여준다.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    import pytz
    import json

    # 1) prediction_log 로딩
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return "<h2>⚠️ prediction_log.csv 파일이 없습니다.</h2>"

    # 🔍 학습용 trainer 행, train* source 행 제거
    df = _filter_prediction_rows(df)

    if df.empty:
        return "<h2>⚠️ 예측 기록이 없습니다.</h2>"

    # 2) timestamp 정리 + 최근 100건만 사용
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[df["timestamp"].notna()]
        df = df.sort_values("timestamp", ascending=False).head(100)
    else:
        df = df.head(100)

    # 3) 전체 성공/실패/대기 통계
    status_col = df.get("status")
    if status_col is not None:
        status_col = status_col.astype(str)
        succ_mask = status_col.isin(["success", "v_success"])
        fail_mask = status_col.isin(["fail", "v_fail"])
        pending_mask = (
            status_col.isna()
            | status_col.eq("")
            | status_col.isin(["pending", "v_pending"])
        )
    else:
        succ_mask = fail_mask = pending_mask = pd.Series([], dtype=bool)

    total = len(df)
    total_succ = int(succ_mask.sum()) if len(succ_mask) else 0
    total_fail = int(fail_mask.sum()) if len(fail_mask) else 0
    total_pend = int(pending_mask.sum()) if len(pending_mask) else 0
    total_eval = total_succ + total_fail
    total_sr = (total_succ / total_eval * 100.0) if total_eval > 0 else None

    # 4) 다음 평가 예정 시각(30분마다 평가)
    KST = pytz.timezone("Asia/Seoul")
    now = datetime.now(KST)

    minute = now.minute
    if minute < 30:
        next_min = 30
        extra_hour = 0
    else:
        next_min = 0
        extra_hour = 1
    next_eval = now.replace(
        minute=next_min, second=0, microsecond=0
    ) + timedelta(hours=extra_hour if next_min == 0 else 0)
    next_eval_str = next_eval.strftime("%Y-%m-%d %H:%M")

    # 5) 공통 스타일 + 전체 요약 카드
    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>YOPO 예측·평가 통합 리포트</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding:20px; background:#f4f6fb; }}
        h1   {{ color:#222; }}
        h2   {{ color:#333; border-bottom:1px solid #ccc; padding-bottom:6px; margin-top:20px; }}
        .summary-global {{
            background:#ffffff;
            padding:14px 18px;
            margin:10px 0 18px 0;
            border-radius:10px;
            border-left:6px solid #4a90e2;
            box-shadow:0 1px 4px rgba(0,0,0,0.08);
            font-size:13px;
            line-height:1.6;
        }}
        .card {{
            background:white;
            padding:8px 10px;
            margin-bottom:8px;
            border-radius:8px;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);
            font-size:12px;
            line-height:1.5;
        }}
        .success {{ color:#0a7b27; font-weight:bold; }}
        .fail    {{ color:#c62828; font-weight:bold; }}
        .pending {{ color:#d47f00; font-weight:bold; }}
        .badge {{
            display:inline-block;
            padding:2px 6px;
            border-radius:6px;
            font-size:10px;
            margin-right:4px;
        }}
        .badge-main   {{ background:#e3f2fd; color:#1565c0; }}
        .badge-shadow {{ background:#eeeeee; color:#555555; }}
        .badge-dist   {{ background:#f3e5f5; color:#6a1b9a; }}
        .badge-failpat{{ background:#ffebee; color:#c62828; }}
        .row-line {{ margin:2px 0; }}
        .key      {{ display:inline-block; width:110px; color:#555; }}
        .value    {{ color:#222; }}
    </style>
</head>
<body>
<h1>📘 YOPO — 예측·평가 통합 리포트 (최근 100개)</h1>
<div class="summary-global">
    <div style="font-size:13px; margin-bottom:6px;">
        <b>최근 예측이 제대로 찍히는지, 평가가 잘 되어 성공/실패로 끝나는지 한 번에 보는 화면입니다.</b>
    </div>
    <div class="row-line"><span class="key">기준 시각</span><span class="value">{now.strftime("%Y-%m-%d %H:%M:%S")} KST</span></div>
    <div class="row-line"><span class="key">최근 예측 수</span><span class="value">{total}건</span></div>
    <div class="row-line"><span class="key">평가 완료</span><span class="value">성공 {total_succ}건 / 실패 {total_fail}건</span></div>
    <div class="row-line"><span class="key">평가 대기</span><span class="value">{total_pend}건</span></div>
"""

    if total_sr is None:
        html += (
            '<div class="row-line"><span class="key">최근 성공률</span>'
            '<span class="value">평가가 아직 부족합니다.</span></div>'
        )
    else:
        html += (
            '<div class="row-line"><span class="key">최근 성공률</span>'
            f'<span class="value">{total_sr:.1f}%</span></div>'
        )

    html += f"""
    <div class="row-line"><span class="key">평가 주기</span>
        <span class="value">30분마다 자동 평가 · 다음 평가 {next_eval_str} KST 예정</span>
    </div>
    <div style="margin-top:6px; font-size:11px; color:#666;">
        ⓘ "성공/실패"는 실제로 캔들이 지나간 뒤 평가된 결과이고,<br>
        "대기"는 아직 결과가 확정되지 않아 기다리는 상태입니다.
    </div>
</div>
"""

    # 6) 실패패턴 조회 함수
    try:
        from failure_db import check_failure_exists
    except Exception:
        def check_failure_exists(*args, **kwargs):
            return False

    # 7) 최근 100건 카드 출력
    sorted_rows = df.sort_values("timestamp", ascending=False).to_dict(orient="records")

    for r in sorted_rows:
        sym = str(r.get("symbol", ""))
        strat = str(r.get("strategy", ""))
        ts = str(r.get("timestamp", ""))
        model = str(r.get("model", "") or "")
        direction = str(r.get("direction", "") or "")
        status = str(r.get("status", "") or "")
        reason = str(r.get("reason", "") or "")
        src = str(r.get("source", "") or "")

        # expected_return / rate / return_value 중 하나 사용
        rv = r.get("return_value", None)
        if rv is None or (isinstance(rv, float) and pd.isna(rv)):
            rv = r.get("rate", None)
        if rv is None or (isinstance(rv, float) and pd.isna(rv)):
            rv = r.get("expected_return", 0)
        try:
            rv = float(rv)
        except Exception:
            rv = 0.0
        rv_pct = rv * 100.0

        # 예측 클래스
        pred_class_val = None
        for cname in ["pred_class", "pred_label", "class", "target_class", "bucket", "bin_index"]:
            if cname in r and r[cname] not in [None, "", "nan", "None"]:
                pred_class_val = r[cname]
                break

        # 메타/섀도우
        is_meta = False
        is_shadow = False
        if "is_meta" in r:
            try:
                is_meta = bool(int(r.get("is_meta", 0)))
            except Exception:
                is_meta = str(r.get("is_meta", "")).lower() in ["1", "true", "yes", "y"]
        if "is_shadow" in r:
            try:
                is_shadow = bool(int(r.get("is_shadow", 0)))
            except Exception:
                is_shadow = str(r.get("is_shadow", "")).lower() in ["1", "true", "yes", "y"]
        if not ("is_meta" in r or "is_shadow" in r):
            # 컬럼이 없으면 source에 shadow라는 단어가 있는지로만 대략 추정
            is_shadow = "shadow" in src.lower()

        # 변동성 예측 여부
        is_vol = status.startswith("v_")

        # 실패패턴 존재 여부
        fail_pat = False
        try:
            if status in ["fail", "v_fail"]:
                fail_pat = bool(check_failure_exists(sym, strat))
        except Exception:
            fail_pat = False

        # 상태 스타일/아이콘/텍스트
        if status in ["success", "v_success"]:
            status_class = "success"
            status_icon = "✅"
            eval_text = "평가 완료 (성공)"
        elif status in ["fail", "v_fail"]:
            status_class = "fail"
            status_icon = "❌"
            eval_text = "평가 완료 (실패)"
        elif status in ["pending", "v_pending", ""]:
            status_class = "pending"
            status_icon = "⏳"
            eval_text = f"평가 대기 중 (다음 자동 평가 {next_eval_str} KST 예상)"
        else:
            status_class = "pending"
            status_icon = "❓"
            eval_text = "기타 상태"

        # 보류(Abstain) 여부
        is_abstain = ("abstain" in reason.lower()) or ("보류" in reason)

        # 메타 선택 이유
        meta_reason = None
        for cname in ["meta_reason", "meta_note"]:
            if cname in r and r[cname] not in [None, "", "nan", "None"]:
                meta_reason = str(r[cname])
                break
        if meta_reason is None and is_meta:
            meta_reason = reason or None

        # 예측 타입 텍스트
        if is_meta:
            pred_type = "메타가 실제로 선택한 예측"
        elif is_shadow:
            pred_type = "섀도우(비교용 예측)"
        else:
            pred_type = "일반 예측"

        # 🔍 하이브리드 / 유사도 note JSON 파싱
        hybrid_info_html = ""
        try:
            raw_note = r.get("note", None)
            note_obj = None
            if raw_note not in [None, "", "nan", "None"]:
                if isinstance(raw_note, str):
                    # JSON 문자열로 저장된 경우
                    try:
                        note_obj = json.loads(raw_note)
                    except Exception:
                        # 이미 dict의 str()로 저장된 경우에도 대비
                        note_obj = None
                elif isinstance(raw_note, dict):
                    note_obj = raw_note

            if isinstance(note_obj, dict):
                pieces = []

                w_sim = note_obj.get("hybrid_w_sim")
                w_prob = note_obj.get("hybrid_w_prob")
                if (w_sim is not None) or (w_prob is not None):
                    try:
                        w_sim_f = float(w_sim) if w_sim is not None else None
                    except Exception:
                        w_sim_f = w_sim
                    try:
                        w_prob_f = float(w_prob) if w_prob is not None else None
                    except Exception:
                        w_prob_f = w_prob
                    pieces.append(
                        f"가중치(prob={w_prob_f:.2f} / sim={w_sim_f:.2f})"
                        if isinstance(w_sim_f, float) and isinstance(w_prob_f, float)
                        else f"가중치(prob={w_prob_f}, sim={w_sim_f})"
                    )

                sim_topk = note_obj.get("sim_topk")
                if sim_topk:
                    pieces.append(f"유사도 top-k: {sim_topk}")

                hy_top3 = note_obj.get("hybrid_probs_top3") or note_obj.get("hybrid_top3")
                if hy_top3:
                    pieces.append(f"하이브리드 상위3: {hy_top3}")

                sim_top3 = note_obj.get("sim_probs_top3")
                if sim_top3:
                    pieces.append(f"유사도 상위3: {sim_top3}")

                adj_top3 = note_obj.get("adjusted_probs_top3")
                if adj_top3:
                    pieces.append(f"가드 적용 후 상위3: {adj_top3}")

                filt_top3 = note_obj.get("filtered_probs_top3")
                if filt_top3:
                    pieces.append(f"필터(±1% 이상) 후 상위3: {filt_top3}")

                chosen_model = note_obj.get("chosen_model")
                if chosen_model:
                    pieces.append(f"선택된 모델: {chosen_model}")

                chosen_reason = note_obj.get("chosen_reason")
                if chosen_reason:
                    pieces.append(f"선택 사유: {chosen_reason}")

                # 아무 필드도 못 뽑았으면 전체 note를 백업으로 보여줌
                if not pieces and note_obj:
                    pieces.append(str(note_obj))

                if pieces:
                    hybrid_info_html = " / ".join(pieces)
        except Exception:
            hybrid_info_html = ""

        html += "<div class='card'>"

        # 상단 뱃지들
        html += "<div style='margin-bottom:4px;'>"
        if is_meta:
            html += "<span class='badge badge-main'>META</span>"
        if is_shadow:
            html += "<span class='badge badge-shadow'>SHADOW</span>"
        if is_vol:
            html += "<span class='badge badge-dist'>변동성</span>"
        if fail_pat:
            html += "<span class='badge badge-failpat'>실패패턴 기록됨</span>"
        html += f"<span class='{status_class}' style='margin-left:6px;'>{status_icon} {status or 'status 없음'}</span>"
        html += "</div>"

        # 본문 정보
        html += f"<div class='row-line'><span class='key'>시각</span><span class='value'>{ts}</span></div>"
        html += (
            f"<div class='row-line'><span class='key'>심볼 / 전략</span>"
            f"<span class='value'>{sym} / {strat}</span></div>"
        )
        html += f"<div class='row-line'><span class='key'>예측 종류</span><span class='value'>{pred_type}</span></div>"
        html += f"<div class='row-line'><span class='key'>모델</span><span class='value'>{model or '-'}</span></div>"
        html += f"<div class='row-line'><span class='key'>방향</span><span class='value'>{direction or '-'}</span></div>"
        html += (
            f"<div class='row-line'><span class='key'>예측 클래스</span>"
            f"<span class='value'>{pred_class_val if pred_class_val is not None else '-'}</span></div>"
        )
        html += (
            f"<div class='row-line'><span class='key'>예상 수익률</span>"
            f"<span class='value'>{rv_pct:.2f}%</span></div>"
        )
        html += (
            f"<div class='row-line'><span class='key'>평가 상태</span>"
            f"<span class='value'>{eval_text}</span></div>"
        )

        if is_abstain:
            html += (
                "<div class='row-line'><span class='key'>보류 여부</span>"
                "<span class='value'>예 (조건이 안 맞아서 메타가 보류)</span></div>"
            )
        else:
            html += (
                "<div class='row-line'><span class='key'>보류 여부</span>"
                "<span class='value'>아니오</span></div>"
            )

        if src:
            html += (
                f"<div class='row-line'><span class='key'>source</span>"
                f"<span class='value'>{src}</span></div>"
            )

        # 메타/사유
        if meta_reason:
            html += (
                f"<div class='row-line'><span class='key'>메타 선택 이유</span>"
                f"<span class='value'>{meta_reason}</span></div>"
            )
        elif reason:
            html += (
                f"<div class='row-line'><span class='key'>사유</span>"
                f"<span class='value'>{reason}</span></div>"
            )

        # 🔍 하이브리드/유사도 상세 정보
        if hybrid_info_html:
            html += (
                f"<div class='row-line'><span class='key'>하이브리드/유사도</span>"
                f"<span class='value'>{hybrid_info_html}</span></div>"
            )

        html += "</div>"

    html += "</body></html>"
    return html


# =========================
# 통합 대시보드 라우트
# =========================
@app.route("/check-log-full", methods=["GET"])
def check_log_full():
    """
    📌 예측 + 평가 상태를 한눈에 보는 통합 리포트.
    - prediction_log.csv 최근 100건 기준
    - 예측이 잘 찍히는지
    - 평가가 잘 되어 성공/실패/보류로 끝나는지
    - 성공/실패/보류 사유가 무엇인지
    를 한 화면에서 확인할 수 있다.
    """
    try:
        html = _render_prediction_eval_dashboard_simple()
        return Response(html, mimetype="text/html; charset=utf-8")
    except Exception as e:
        return f"❌ 오류: {e}", 500


@app.route("/check-eval-log", methods=["GET"])
def check_eval_log():
    """
    예전 /check-eval-log 링크를 위한 호환용 라우트.
    이제는 /check-log-full 과 같은 통합 대시보드를 그대로 보여준다.
    """
    try:
        html = _render_prediction_eval_dashboard_simple()
        return Response(html, mimetype="text/html; charset=utf-8")
    except Exception as e:
        return f"❌ 오류: {e}", 500


@app.route("/diag/e2e")
def diag_e2e():
    try:
        group = int(request.args.get("group", "-1"))
        view = request.args.get("view", "json").lower()
        cumulative = request.args.get("cum", "0") == "1"
        symbols = request.args.get("symbols")
        out = diag_e2e_run(group=group, view=view, cumulative=cumulative, symbols=symbols)
        if isinstance(out, Response):
            return out
        if view == "html":
            return Response(
                out if isinstance(out, str) else str(out),
                mimetype="text/html; charset=utf-8",
            )
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/run")
def run():
    try:
        if (
            os.path.exists(LOCK_PATH)
            or _is_training()
            or _is_group_active_file()
            or os.path.exists(GROUP_TRAIN_LOCK)
        ):
            return "⏸️ 학습/초기화/그룹학습 진행 중: 예측 시작 차단됨", 423
        print("[RUN] 전략별 예측 실행")
        sys.stdout.flush()
        _pl_clear()
        _safe_open_gate("route_run")
        try:
            for strategy in ["단기", "중기", "장기"]:
                main(strategy, force=True)
        finally:
            _safe_close_gate("route_run")
        return "Recommendation started"
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}", 500


@app.route("/train-now")
def train_now():
    try:
        if os.path.exists(LOCK_PATH):
            return "⏸️ 초기화 중: 학습 시작 차단됨", 423
        force = request.args.get("force", "0") == "1"
        _safe_close_gate("train_now_start")
        started = _start_train_loop_safe(force_restart=force, sleep_sec=0)
        if started:
            return "✅ 전체 그룹 학습 루프 시작됨 (백그라운드)"
        return (
            "⏳ 이미 실행 중 (재가동 생략)"
            if not force
            else "⏳ 재가동 시도했으나 기존 루프 유지됨"
        )
    except Exception as e:
        return f"학습 실패: {e}", 500

@app.route("/train-log")
def train_log():
    """
    📈 학습 로그 보기 (아주 쉬운 버전)

    - 각 심볼/전략이 얼마나 잘 학습되었는지
    - 정확도 / F1 / loss 를
    카드처럼 쉽게 보여준다.
    """
    try:
        # logger.py 에서 만든 카드 데이터 불러오기
        cards = get_train_log_cards(max_cards=200)
        log_path = TRAIN_LOG if isinstance(TRAIN_LOG, str) else ""

        # 1) 아직 학습 기록이 전혀 없는 경우
        if not cards:
            return f"""
<html>
<head>
    <meta charset="utf-8">
    <title>YOPO 학습 로그</title>
</head>
<body style="font-family:Arial, sans-serif;background:#f4f6fb;padding:20px;font-size:14px;">
    <h1>📘 YOPO — 학습 로그</h1>
    <div style="background:#fff;padding:14px 18px;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.08);">
        <div style="font-weight:bold;margin-bottom:6px;">아직 저장된 학습 결과가 없습니다.</div>
        <div style="font-size:13px;color:#555;">
            ▶ 학습이 한 번이라도 끝나면 이 화면에<br>
            &nbsp;&nbsp;&nbsp;심볼별 카드가 자동으로 생깁니다.<br><br>
            <small>기록 파일 위치: <code>{log_path}</code></small>
        </div>
    </div>
</body>
</html>
"""

        # 2) 전체 개수/OK 개수/점검 필요 개수
        total_cards = len(cards)
        ok_cards = sum(1 for c in cards if str(c.get("health", "")).upper() == "OK")
        bad_cards = total_cards - ok_cards

        # 3) 가장 최근 학습 1건 요약
        try:
            cards_sorted = sorted(
                cards,
                key=lambda c: str(c.get("timestamp", "")) or ""
            )
            last = cards_sorted[-1]
        except Exception:
            last = cards[-1]

        last_sym = last.get("symbol", "알 수 없음")
        last_strat = last.get("strategy", "알 수 없음")
        last_model = last.get("model", "") or "알 수 없음"
        last_ts = last.get("timestamp", "알 수 없음")

        def _to_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        last_acc = _to_float(last.get("val_acc", 0.0))
        last_f1 = _to_float(last.get("val_f1", 0.0))
        last_loss = _to_float(last.get("val_loss", 0.0))
        last_health_text = last.get("health_text", "상태 정보 없음")

        # 4) 카드 하나씩 HTML 만들기 (아주 단순한 문장만 사용)
        card_blocks = []
        for c in cards:
            sym = c.get("symbol", "")
            strat = c.get("strategy", "")
            model = c.get("model", "") or "알 수 없음"

            acc = _to_float(c.get("val_acc", 0.0))
            f1 = _to_float(c.get("val_f1", 0.0))
            loss = _to_float(c.get("val_loss", 0.0))

            health_text = c.get("health_text", "상태 정보 없음")
            status = c.get("status", "")

            ts = c.get("timestamp", "알 수 없음")

            data_summary = c.get("data_summary", "")
            ret_summary = c.get("ret_summary_text", "수익률 분포 정보 없음")
            coverage_summary = c.get("coverage_summary", "검증 커버리지 정보 없음")

            block = f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:10px 12px;margin-bottom:10px;background:#ffffff;">
  <div style="font-weight:bold;margin-bottom:4px;">
    {sym} · {strat}
    <span style="font-size:11px;color:#777;"> (모델: {model})</span>
  </div>

  <div style="font-size:12px;margin-bottom:4px;">
    ● 이 조합의 학습 결과입니다.
  </div>

  <div style="font-size:12px;margin-bottom:2px;">
    ▷ 정확도(정답 잘 맞춘 비율): <b>{acc:.4f}</b>
  </div>
  <div style="font-size:12px;margin-bottom:2px;">
    ▷ F1 점수(정답·오답 균형): <b>{f1:.4f}</b>
  </div>
  <div style="font-size:12px;margin-bottom:6px;">
    ▷ loss(작을수록 좋음): <b>{loss:.4f}</b>
  </div>

  <div style="font-size:12px;margin-bottom:2px;color:#b71c1c;">
    ● 상태 요약: {health_text} {(' (status=' + status + ')') if status else ''}
  </div>

  <div style="font-size:11px;margin-top:4px;color:#333;">
    ● 데이터 양: {data_summary}
  </div>
  <div style="font-size:11px;margin-top:2px;color:#333;">
    ● 수익률 분포: {ret_summary}
  </div>
  <div style="font-size:11px;margin-top:2px;color:#333;">
    ● 검증 커버리지: {coverage_summary}
  </div>

  <div style="font-size:11px;color:#777;margin-top:4px;">
    마지막 학습 시간: {ts}
  </div>
</div>
"""
            card_blocks.append(block)

        class_cards_html = "\n".join(card_blocks)

        # 5) 최종 HTML
        html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>YOPO 학습 로그 (아주 쉬운 버전)</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background:#f4f6fb;
            padding:20px;
            font-size:14px;
        }}
        h1 {{ color:#222; }}
        .card {{
            background:#ffffff;
            padding:14px 18px;
            margin-bottom:16px;
            border-radius:10px;
            box-shadow:0 1px 4px rgba(0,0,0,0.08);
            line-height:1.6;
        }}
    </style>
</head>
<body>
<h1>📘 YOPO — 학습 로그 (쉽게 보기)</h1>

<div class="card">
    <div style="font-weight:bold;margin-bottom:6px;">1️⃣ 지금까지 학습된 전체 요약</div>
    <div>· 총 <b>{total_cards}</b>개 심볼·전략 조합이 학습되었습니다.</div>
    <div>· 이 중 <b>{ok_cards}</b>개는 <span style="color:#2e7d32;">정상(OK)</span>, <b>{bad_cards}</b>개는 <span style="color:#b71c1c;">추가 확인 필요</span>입니다.</div>
    <div style="font-size:12px;color:#666;margin-top:4px;">
        기록 파일: <code>{log_path}</code>
    </div>
</div>

<div class="card">
    <div style="font-weight:bold;margin-bottom:6px;">2️⃣ 가장 최근에 끝난 학습 한 줄 요약</div>
    <div>· 시간: <b>{last_ts}</b></div>
    <div>· 심볼 / 전략: <b>{last_sym} / {last_strat}</b></div>
    <div>· 모델: <b>{last_model}</b></div>
    <div>· 정확도: <b>{last_acc:.4f}</b> / F1: <b>{last_f1:.4f}</b> / loss: <b>{last_loss:.4f}</b></div>
    <div>· 상태: {last_health_text}</div>
</div>

<div class="card">
    <div style="font-weight:bold;margin-bottom:6px;">3️⃣ 심볼·전략별 자세한 카드</div>
    <div style="font-size:13px;color:#555;margin-bottom:6px;">
        각 카드 하나가 “심볼 + 전략(단기/중기/장기)”의 학습 결과입니다.<br>
        위에서부터 차근차근 읽으면 어느 부분이 좋은지, 어느 부분을 더 키워야 할지 쉽게 볼 수 있습니다.
    </div>
    {class_cards_html}
</div>

</body>
</html>
"""
        return html

    except Exception as e:
        return f"읽기 오류: {e}", 500




@app.route("/models")
def list_models():
    try:
        files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
        return "<pre>" + "\n".join(files) + "</pre>" if files else "models 폴더 비어 있음"
    except Exception as e:
        return f"오류: {e}", 500


@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG):
            return jsonify({"error": "prediction_log.csv 없음"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = _filter_prediction_rows(df)
        if "timestamp" not in df:
            return jsonify([])
        df = df[df["timestamp"].notna()]
        return jsonify(df.tail(10).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/train-symbols", methods=["GET", "POST"])
def train_symbols():
    try:
        if os.path.exists(LOCK_PATH):
            return "⏸️ 초기화 중: 그룹/선택 학습 시작 차단됨", 423

        def _ensure_single_loop(force_flag: bool):
            if _is_training():
                if not force_flag:
                    return False, (
                        "🚫 이미 메인 학습 루프 실행 중 "
                        "(force=1 또는 force=true 로 강제 교체 가능)",
                        409,
                    )
                try:
                    _request_stop_safe()
                    _stop_train_loop_safe(timeout=45)
                except Exception:
                    pass
            return True, None

        if request.method == "GET":
            group_idx = int(request.args.get("group", -1))
            force = request.args.get("force", "0") == "1"
            if group_idx < 0 or group_idx >= len(SYMBOL_GROUPS):
                return f"❌ 잘못된 그룹 번호: {group_idx}", 400
            ok, resp = _ensure_single_loop(force)
            if not ok:
                return resp
            group_symbols = SYMBOL_GROUPS[group_idx]
            print(f"🚀 그룹 학습 요청됨 → 그룹 #{group_idx} | 심볼: {group_symbols}")
            # ✅ 학습 시작하기 전에 최신 캔들 강제 수집
            _warmup_latest_klines(group_symbols)
            # 그룹 시작: 게이트 닫기 + GROUP_ACTIVE 생성 + 그룹학습 락 생성
            _safe_close_gate("train_group_start")
            _set_group_active(True, group_idx=group_idx, symbols=group_symbols)
            try:
                with open(GROUP_TRAIN_LOCK, "w", encoding="utf-8") as f:
                    f.write(
                        f"group={group_idx} started={datetime.datetime.utcnow().isoformat()}\n"
                    )
                print("[GROUP-LOCK] created")
                sys.stdout.flush()
            except Exception as e:
                print(f"[GROUP-LOCK] create failed: {e}")
                sys.stdout.flush()

            def _worker():
                try:
                    _train_models_safe(group_symbols)
                    if not group_all_complete():
                        print("[GROUP-AFTER] 미완료: group_all_complete()=False → 예측 생략")
                        return
                    if not ready_for_group_predict():
                        print("[GROUP-AFTER] 미완료: ready_for_group_predict()=False → 예측 생략")
                        return
                    _predict_after_training(group_symbols, source_note=f"group{group_idx}_after_train")
                    try:
                        mark_group_predicted()
                        print("[GROUP-AFTER] mark_group_predicted() 호출 완료")
                    except Exception as e:
                        print(f"[GROUP-AFTER] mark_group_predicted 예외: {e}")
                finally:
                    # 종료 시점에만 GROUP_ACTIVE 삭제 + 그룹학습 락 제거
                    if group_all_complete() and ready_for_group_predict():
                        _set_group_active(False)
                    try:
                        if os.path.exists(GROUP_TRAIN_LOCK):
                            os.remove(GROUP_TRAIN_LOCK)
                            print("[GROUP-LOCK] removed")
                            sys.stdout.flush()
                    except Exception as e:
                        print(f"[GROUP-LOCK] remove failed: {e}")
                        sys.stdout.flush()

            threading.Thread(target=_worker, daemon=True).start()
            return (
                f"✅ 그룹 #{group_idx} 학습 시작됨 "
                "(완료 검증 통과 시 학습 직후 예측, 이후 mark_group_predicted)"
            )
        else:
            body = request.get_json(silent=True) or {}
            symbols = body.get("symbols", [])
            force = bool(body.get("force", False))
            if not isinstance(symbols, list) or not symbols:
                return "❌ 유효하지 않은 symbols 리스트", 400
            ok, resp = _ensure_single_loop(force)
            if not ok:
                return resp
            # ✅ 선택 학습도 마찬가지로 시작 전에 최신 캔들 강제 수집
            _warmup_latest_klines(symbols)
            # 선택 학습은 그룹 경계 아님 → GROUP_ACTIVE 비조작, 그룹 락 비사용
            _safe_close_gate("train_selected_start")

            def _worker_sel():
                try:
                    _train_models_safe(symbols)
                    _predict_after_training(symbols, source_note="selected_after_train")
                finally:
                    pass

            threading.Thread(target=_worker_sel, daemon=True).start()
            return f"✅ {len(symbols)}개 심볼 학습 시작됨 (학습 직후 예측 수행 — 그룹 마킹 없음)"
    except Exception as e:
        traceback.print_exc()
        return f"❌ 오류: {e}", 500


@app.route("/meta-fix-now")
def meta_fix_now():
    try:
        maintenance_fix_meta.fix_all_meta_json()
        return "✅ meta.json 점검/복구 완료"
    except Exception as e:
        return f"⚠️ 실패: {e}", 500


# 🔴 reset-all 라우트 (봇 차단 + 풀초기화 + 바로 재시작)
@app.route("/reset-all", methods=["GET", "POST"])
@app.route("/reset-all/<key>", methods=["GET", "POST"])
def reset_all(key=None):
    # 0) 카카오톡/페북 미리보기 같은 봇은 차단
    ua_raw = request.headers.get("User-Agent", "")
    ua = ua_raw.lower()
    if "facebookexternalhit" in ua or "kakaotalk-scrap" in ua:
        return "❌ bot blocked", 403

    # 1) 인증키 체크
    req_key = key or request.args.get("key") or (request.json.get("key") if request.is_json else None)
    if req_key != "3572":
        print(f"[RESET] 인증 실패 from {request.remote_addr} path={request.path}")
        sys.stdout.flush()
        return "❌ 인증 실패", 403

    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    print(f"[RESET] 요청 수신 from {ip} UA={ua_raw}")
    sys.stdout.flush()

    # 리셋 중 예측 막기
    _safe_close_gate("reset_enter")

    def _do_reset_work():
        stop_timeout = int(os.getenv("RESET_STOP_TIMEOUT", "12"))
        max_wait = int(os.getenv("RESET_MAX_WAIT_SEC", "120"))
        poll_sec = max(1, int(os.getenv("RESET_POLL_SEC", "2")))
        qwipe_early = os.getenv("RESET_QWIPE_EARLY", "1") == "1"

        try:
            from data.utils import _kline_cache, _feature_cache
        except Exception:
            _kline_cache = type("dummy", (), {"clear": lambda self: None})()
            _feature_cache = type("dummy", (), {"clear": lambda self: None})()

        # 1. 전역락 획득
        _acquire_global_lock()

        # 2. 스케줄러/cleanup 중단
        _stop_all_aux_schedulers()
        _pl_clear()

        print("[RESET] 백그라운드 초기화 시작")
        sys.stdout.flush()

        # 3. 학습 루프 정지 요청
        try:
            if hasattr(train, "request_stop"):
                _request_stop_safe()
        except Exception:
            pass

        # 3-0. 빠른 정지 시도
        print(f"[RESET] 학습 루프 정지 시도(timeout={stop_timeout}s)")
        sys.stdout.flush()
        stopped = _stop_train_loop_safe(timeout=stop_timeout)
        print(f"[RESET] stop_train_loop 결과: {stopped}")
        sys.stdout.flush()

        # 3-1. 그래도 안 멈추면, 옵션에 따라 조기 QWIPE
        if (not stopped) and qwipe_early:
            print("[RESET] 학습루프 멈춤 실패 → QWIPE 강제 1회 실행")
            sys.stdout.flush()
            try:
                _quarantine_wipe_persistent()
                ensure_dirs()
            except Exception as e:
                print(f"[RESET] QWIPE 실패: {e}")

        # 3-2. max_wait 만큼 추가 대기
        if not stopped:
            t0 = time.time()
            print(f"[RESET] 정지 대기 시작… 최대 {max_wait}s (폴링 {poll_sec}s)")
            sys.stdout.flush()
            while time.time() - t0 < max_wait:
                try:
                    if not _is_training():
                        stopped = True
                        break
                except Exception:
                    pass
                try:
                    if _stop_train_loop_safe(timeout=2):
                        stopped = True
                        break
                except Exception:
                    pass
                time.sleep(poll_sec)
            print(f"[RESET] 정지 대기 완료 → stopped={stopped}")
            sys.stdout.flush()

        # 4. PERSIST_DIR 완전 초기화
        try:
            # 모델/로그/ssl_models 제거 후 재생성
            for d in [MODEL_DIR, LOG_DIR, os.path.join(PERSIST_DIR, "ssl_models")]:
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)

            # 남은 파일 중 LOCK_DIR 제외하고 모두 삭제
            keep = {os.path.basename(LOCK_DIR)}
            for name in list(os.listdir(PERSIST_DIR)):
                if name in keep:
                    continue
                p = os.path.join(PERSIST_DIR, name)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            # 필수 폴더/로그 다시 생성
            ensure_dirs()
            ensure_prediction_log_exists()
            ensure_train_log_exists()
        except Exception as e:
            print(f"[RESET] 풀와이프 예외: {e}")
            sys.stdout.flush()

        # 5. 캐시 비우기
        try:
            _kline_cache.clear()
        except Exception:
            pass
        try:
            _feature_cache.clear()
        except Exception:
            pass

        # 6. cleanup + scheduler 재시작
        try:
            start_cleanup_scheduler()
        except Exception as e:
            print(f"[RESET] cleanup 재시작 실패: {e}")

        try:
            start_scheduler()
        except Exception as e:
            print(f"[RESET] 스케줄러 재시작 실패: {e}")

        # 7. 학습 루프 재시작
        try:
            _safe_close_gate("reset_done_reopen")
            started = _start_train_loop_safe(force_restart=True, sleep_sec=0)
            print(f"[RESET] 학습 재시작 결과: {started}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[RESET] 학습 재시작 실패: {e}")

        print("🔚 [RESET] 정리 + 재시작 완료")
        sys.stdout.flush()
        _release_global_lock()

    threading.Thread(target=_do_reset_work, daemon=True).start()

    return Response(
        "✅ 초기화 요청 접수됨. 정리 후 바로 학습/스케줄러를 다시 켭니다.\n"
        "로그에서 [RESET] 태그를 확인하세요.",
        mimetype="text/plain; charset=utf-8",
    )


@app.route("/force-fix-prediction_log")
@app.route("/force-fix-prediction-log")
def force_fix_prediction_log():
    try:
        if os.path.exists(PREDICTION_LOG):
            os.remove(PREDICTION_LOG)
        ensure_prediction_log_exists()
        print("[FORCE-FIX] prediction_log.csv 재생성 완료")
        sys.stdout.flush()
        return "✅ prediction_log.csv 강제 초기화 완료"
    except Exception as e:
        return f"⚠️ 오류: {e}", 500


# 즉시 전체 예측: 학습 일시정지 → 예측 → 학습 재개
@app.route("/predict-now", methods=["POST", "GET"])
def predict_now():
    try:
        was_running = _is_training()
        if was_running:
            print("[PREDICT-NOW] training detected → stopping...")
            sys.stdout.flush()
            try:
                _request_stop_safe()
            except Exception:
                pass
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
                for strat in ["단기", "중기", "장기"]:
                    total += 1
                    if not has_model_for(sym, strat):
                        skipped += 1
                        print(f"[PREDICT-NOW] skip {sym}-{strat}: model missing")
                        sys.stdout.flush()
                        continue
                    try:
                        result = predict(sym, strat, source="predict_now", model_type=None)
                        if not isinstance(result, dict):
                            try:
                                from predict import failed_result

                                failed_result(
                                    sym,
                                    strat,
                                    reason="invalid_return",
                                    source="predict_now",
                                )
                            except Exception:
                                pass
                        done += 1
                        print(f"[PREDICT-NOW] ok {sym}-{strat}")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"[PREDICT-NOW] fail {sym}-{strat}: {e}")
                        sys.stdout.flush()
                        try:
                            from predict import failed_result

                            failed_result(sym, strat, reason=str(e), source="predict_now")
                        except Exception:
                            pass
        finally:
            _safe_close_gate("predict_now_end")

        resumed = False
        if was_running:
            resumed = _start_train_loop_safe(force_restart=False, sleep_sec=0)
            print(f"[PREDICT-NOW] training resumed={resumed}")
            sys.stdout.flush()

        return (
            f"✅ 예측 완료 | 총:{total} 성공:{done} 스킵:{skipped} | "
            f"학습정지:{'예' if was_running else '아니오'} → 재개:{'예' if resumed else '아니오'}"
        )
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
