# app.py — single-source, deduped train loop via train.py (ONE concurrent loop only)

from flask import Flask, jsonify, request, Response
from recommend import main
import train, os, threading, datetime, pytz, traceback, sys, shutil, re, time  # time 사용
import pandas as pd  # ← ✅ 별칭 임포트는 단독 줄로 분리해야 문법 오류 없음
from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_trigger import run as trigger_run
from data.utils import SYMBOLS, get_kline_by_strategy
from visualization import generate_visual_report, generate_visuals_for_strategy
from wrong_data_loader import load_training_prediction_data
from predict import evaluate_predictions
from train import train_symbol_group_loop  # (호환용) 직접 호출 루트 남김
import maintenance_fix_meta
from logger import ensure_prediction_log_exists

# 👇 무결성 점검(있으면 실행)
try:
    from integrity_guard import run as _integrity_check
    _integrity_check()
except Exception as e:
    print(f"[WARN] integrity_guard skipped: {e}")

# ✅ 종합점검 모듈(HTML/JSON + 누적 통계 지원)
from diag_e2e import run as diag_e2e_run

# ✅ cleanup 모듈 경로 보정
try:
    from scheduler_cleanup import start_cleanup_scheduler   # [KEEP]
    import safe_cleanup                                      # [KEEP]
    import scheduler_cleanup as _cleanup_mod                 # 🆕 stop 지원용 참조
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scheduler_cleanup import start_cleanup_scheduler    # [KEEP]
    import safe_cleanup                                      # [KEEP]
    import scheduler_cleanup as _cleanup_mod                 # 🆕

# ===== 경로 통일 =====
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))  # ← 루트 탐색용(초기화 강화에만 사용)
PERSIST_DIR= "/persistent"
LOG_DIR    = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR  = os.path.join(PERSIST_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)  # ✅ 모델 디렉터리 보장

# ===== 글로벌 락 유틸(전체 일시정지) =====
LOCK_DIR   = getattr(safe_cleanup, "LOCK_DIR", os.path.join(PERSIST_DIR, "locks"))
LOCK_PATH  = getattr(safe_cleanup, "LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))
os.makedirs(LOCK_DIR, exist_ok=True)

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

# ---------- 🆕 공통 유틸: 즉시 격리-와이프 ----------
def _quarantine_wipe_persistent():
    """
    /persistent 내부를 통째로 비우되, 충돌을 피하기 위해
    내용을 /persistent/_trash_<ts>/ 로 **원자적으로 이동** 후
    깨끗한 기본 디렉터리 구조를 재생성한다.
    """
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trash_dir = os.path.join(PERSIST_DIR, f"_trash_{ts}")
    os.makedirs(trash_dir, exist_ok=True)
    keep_names = {os.path.basename(LOCK_DIR)}  # 락 디렉터리는 유지

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
    # 깨끗한 기본 구조 재생성
    for d in ["logs", "models", "ssl_models"]:
        os.makedirs(os.path.join(PERSIST_DIR, d), exist_ok=True)

    print(f"🧨 [QWIPE] moved_to_trash={moved} trash_dir={trash_dir}"); sys.stdout.flush()
    return trash_dir

# 🆘 DB/SQLite 열기 전, 무조건 1회 응급 정리(락/보호시간 무시) → 백그라운드 실행으로 변경
def _async_emergency_purge():
    try:
        # 먼저 트래시 디렉터리들 제거 (이전 리셋 잔여물 정리)
        try:
            for name in list(os.listdir(PERSIST_DIR)):
                if name.startswith("_trash_"):
                    path = os.path.join(PERSIST_DIR, name)
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"[BOOT-CLEANUP] trashed removed: {name}")
        except Exception as e:
            print(f"⚠️ [BOOT-CLEANUP] trash 제거 실패: {e}")

        # 하드캡 초과 시에만 EMERGENCY, 그 외에는 옵션에 따라 온건 정리 또는 아무것도 안 함
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

# ✅ prediction_log은 logger와 동일한 위치/헤더로 관리
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")

LOG_FILE          = os.path.join(LOG_DIR, "train_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG         = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG       = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_LOG       = os.path.join(LOG_DIR, "failure_count.csv")

# ✅ 서버 시작 직전 용량 정리 (환경변수로 제어)
try:
    if os.getenv("CLEANUP_ON_BOOT", "0") == "1":
        print("[BOOT-CLEANUP] CLEANUP_ON_BOOT=1 → logs/models 정리 시작"); sys.stdout.flush()
        safe_cleanup.cleanup_logs_and_models()
        print("[BOOT-CLEANUP] 완료"); sys.stdout.flush()
    else:
        print("[BOOT-CLEANUP] 비활성화(CLEANUP_ON_BOOT=0)"); sys.stdout.flush()
except Exception as e:
    print(f"[경고] startup cleanup 실패: {e}"); sys.stdout.flush()

# ✅ 로그 파일 존재 보장(정확 헤더)
try:
    from logger import ensure_train_log_exists
    ensure_train_log_exists()
except Exception:
    pass
ensure_prediction_log_exists()

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# -----------------------------
# 스케줄러 (평가/트리거/메타복구)
# -----------------------------
_sched = None
def start_scheduler():
    global _sched
    if _sched is not None:
        print("⚠️ 스케줄러 이미 실행 중, 재시작 생략"); sys.stdout.flush()
        return

    # ✅ 리셋 중이면 시작 금지
    if os.path.exists(LOCK_PATH):
        print("⏸️ 리셋 락 감지 → 스케줄러 시작 지연"); sys.stdout.flush()
        return

    print(">>> 스케줄러 시작"); sys.stdout.flush()
    sched = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))

    # ✅ 전략별 평가(30분마다) — 실행 함수를 직접 등록해야 함
    def 평가작업(strategy):
        def wrapped():
            try:
                ts = now_kst().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[EVAL][{ts}] 전략={strategy} 시작"); sys.stdout.flush()
                evaluate_predictions(lambda sym, _: get_kline_by_strategy(sym, strategy))
            except Exception as e:
                print(f"[EVAL] {strategy} 실패: {e}")
        return wrapped

    for strat in ["단기", "중기", "장기"]:
        # ⛏️ 버그수정: 이전 코드(lambda s=strat: 평가작업(s))는 callable을 반환하지 않아 실행 안 됨
        sched.add_job(평가작업(strat), trigger="interval", minutes=30,
                      id=f"eval_{strat}", replace_existing=True)

    # ✅ 예측 트리거(메타적용 포함) 30분
    sched.add_job(trigger_run, "interval", minutes=30, id="predict_trigger", replace_existing=True)

    # ✅ 메타 JSON 정합성/복구 주기작업 (30분)
    def meta_fix_job():
        try:
            maintenance_fix_meta.fix_all_meta_json()
        except Exception as e:
            print(f"[META-FIX] 주기작업 실패: {e}")
    sched.add_job(meta_fix_job, "interval", minutes=30, id="meta_fix", replace_existing=True)

    sched.start()
    _sched = sched
    print("✅ 스케줄러 시작 완료"); sys.stdout.flush()

def _pause_and_clear_scheduler():
    """초기화 동안 스케줄러 완전 정지(작업 제거)"""
    global _sched
    try:
        if _sched is not None:
            print("[SCHED] pause + remove_all_jobs"); sys.stdout.flush()
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
        print(f"[SCHED] 정지 실패: {e}"); sys.stdout.flush()

# 🆕 Cleanup 스케줄러 및 잠재 스케줄러까지 전부 끄기
def _stop_all_aux_schedulers():
    try:
        # 1) 앱 내부 스케줄러
        _pause_and_clear_scheduler()
    except Exception:
        pass
    try:
        # 2) cleanup 모듈 내 스케줄러
        if hasattr(_cleanup_mod, "stop_cleanup_scheduler"):
            try:
                _cleanup_mod.stop_cleanup_scheduler()
                print("🧹 [SCHED] cleanup 스케줄러 stop 호출"); sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ cleanup stop 실패: {e}"); sys.stdout.flush()
        # fallback: 모듈 속 BackgroundScheduler 탐색 후 shutdown
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

# ===== Flask =====
app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

# -----------------------------
# 백그라운드 초기화(한 번만)
# -----------------------------
_INIT_DONE = False
_INIT_LOCK = threading.Lock()

def _init_background_once():
    global _INIT_DONE
    with _INIT_LOCK:
        if _INIT_DONE:
            return
        try:
            # 리셋 중이면 모든 백그라운드 시작 금지
            if os.path.exists(LOCK_PATH):
                print("⏸️ 락 감지 → 백그라운드 초기화 지연"); sys.stdout.flush()
                return

            from failure_db import ensure_failure_db
            print(">>> 서버 실행 준비")
            ensure_failure_db(); print("✅ failure_patterns DB 초기화 완료")

            # 학습 루프 스레드 — train.py의 단일 루프 보장 API 사용
            train.start_train_loop(force_restart=False, sleep_sec=0)
            print("✅ 학습 루프 스레드 시작")

            # 정리 스케줄러(기본 30분)
            start_cleanup_scheduler()
            print("✅ cleanup 스케줄러 시작")

            # 평가/트리거/메타복구 스케줄러
            try:
                start_scheduler()
            except Exception as e:
                print(f"⚠️ 스케줄러 시작 실패: {e}")

            # 메타 보정(부팅 시 1회 선 실행)
            threading.Thread(target=maintenance_fix_meta.fix_all_meta_json, daemon=True).start()
            print("✅ maintenance_fix_meta 초기 실행 트리거")

            # 텔레그램 알림
            try:
                send_message("[시작] YOPO 서버 실행됨")
                print("✅ Telegram 알림 발송 완료")
            except Exception as e:
                print(f"⚠️ Telegram 발송 실패: {e}")

            _INIT_DONE = True
            print("✅ 백그라운드 초기화 완료")
        except Exception as e:
            print(f"❌ 백그라운드 초기화 실패: {e}")

# Flask 3.1 호환
if hasattr(app, "before_serving"):
    @app.before_serving
    def _boot_once():
        _init_background_once()
else:
    @app.before_request
    def _boot_once_compat():
        if not _INIT_DONE:
            _init_background_once()

# ===== 라우트 =====
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

    # 모델 파일 파싱 (.pt / .ptz / .safetensors 모두)
    try:
        model_files = [f for f in os.listdir(MODEL_DIR)
                       if f.endswith((".pt", ".ptz", ".safetensors"))]
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
                pred["return"] = pd.to_numeric(pred.get("return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
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

@app.route("/")
def index(): return "Yopo server is running"

@app.route("/ping")
def ping(): return "pong"

# ✅ 종합 점검 라우트 (HTML/JSON + 누적 옵션)
@app.route("/diag/e2e")
def diag_e2e():
    """
    사용법:
      /diag/e2e?view=json         → JSON(기본)
      /diag/e2e?view=html         → 한글 HTML 리포트
      /diag/e2e?group=0           → 그룹 인덱스 기준 통계
      /diag/e2e?cum=1             → 누적 통계(메모리 안전 스트리밍)
      /diag/e2e?symbols=BTCUSDT,ETHUSDT → 특정 심볼만 집계
    """
    try:
        group = int(request.args.get("group", "-1"))
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
        if os.path.exists(LOCK_PATH):
            return "⏸️ 초기화 중: 예측 시작 차단됨", 423
        print("[RUN] 전략별 예측 실행"); sys.stdout.flush()
        for strategy in ["단기","중기","장기"]:
            main(strategy, force=True)
        return "Recommendation started"
    except Exception as e:
        traceback.print_exc(); return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    """쿼리 force=1이면 강제 재가동, 아니면 안전 시작(이미 실행 중이면 스킵 메시지)."""
    try:
        # 리셋 중이면 시작 금지
        if os.path.exists(LOCK_PATH):
            return "⏸️ 초기화 중: 학습 시작 차단됨", 423
        force = request.args.get("force", "0") == "1"
        started = train.start_train_loop(force_restart=force, sleep_sec=0)
        if started:
            return "✅ 전체 그룹 학습 루프 시작됨 (백그라운드)"
        else:
            return "⏳ 이미 실행 중 (재가동 생략)" if not force else "⏳ 재가동 시도했으나 기존 루프 유지됨"
    except Exception as e:
        return f"학습 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        if not os.path.exists(LOG_FILE): return "학습 로그 없음"
        df = pd.read_csv(LOG_FILE, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty or df.shape[1] == 0: return "학습 기록 없음"
        return "<pre>" + df.to_csv(index=False) + "</pre>"
    except Exception as e:
        return f"읽기 오류: {e}", 500

@app.route("/models")
def list_models():
    try:
        if not os.path.exists(MODEL_DIR): return "models 폴더 없음"
        files = os.listdir(MODEL_DIR)
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
            html += f"<tr><td>{r.get('timestamp','')}</td><td>{r.get('symbol','')}</td><td>{r.get('strategy','')}</td><td>{r.get('model','')}</td><td>{r.get('direction','')}</td><td>{float(r.get('return',0) or 0):.4f}</td><td>{icon}</td><td>{r.get('reason','')}</td></tr>"
        html += "</table>"
        return html
    except Exception as e:
        return f"❌ 오류: {e}", 500

from data.utils import SYMBOL_GROUPS

@app.route("/train-symbols")
def train_symbols():
    try:
        if os.path.exists(LOCK_PATH):
            return f"⏸️ 초기화 중: 그룹 학습 시작 차단됨", 423

        group_idx = int(request.args.get("group", -1))
        force = request.args.get("force", "0") == "1"
        if group_idx < 0 or group_idx >= len(SYMBOL_GROUPS):
            return f"❌ 잘못된 그룹 번호: {group_idx}", 400
        group_symbols = SYMBOL_GROUPS[group_idx]

        # 단일 루프 보장
        if train.is_loop_running():
            if not force:
                return "🚫 이미 메인 학습 루프 실행 중 (force=1 로 강제 교체 가능)", 409
            train.stop_train_loop(timeout=45)

        print(f"🚀 그룹 학습 요청됨 → 그룹 #{group_idx} | 심볼: {group_symbols}")
        threading.Thread(target=lambda: train.train_models(group_symbols), daemon=True).start()
        return f"✅ 그룹 #{group_idx} 학습 시작됨 (단일 루프 보장)"
    except Exception as e:
        traceback.print_exc(); return f"❌ 오류: {e}", 500

@app.route("/train-symbols", methods=["POST"])
def train_selected_symbols():
    try:
        if os.path.exists(LOCK_PATH):
            return "⏸️ 초기화 중: 선택 학습 시작 차단됨", 423

        body = request.get_json(silent=True) or {}
        symbols = body.get("symbols", [])
        force = bool(body.get("force", False))
        if not isinstance(symbols, list) or not symbols:
            return "❌ 유효하지 않은 symbols 리스트", 400

        if train.is_loop_running():
            if not force:
                return "🚫 이미 메인 학습 루프 실행 중 (force=true 로 강제 교체 가능)", 409
            train.stop_train_loop(timeout=45)

        threading.Thread(target=lambda: train.train_models(symbols), daemon=True).start()
        return f"✅ {len(symbols)}개 심볼 학습 시작됨 (단일 루프 보장)"
    except Exception as e:
        return f"❌ 학습 실패: {e}", 500

@app.route("/meta-fix-now")
def meta_fix_now():
    try:
        maintenance_fix_meta.fix_all_meta_json()
        return "✅ meta.json 점검/복구 완료"
    except Exception as e:
        return f"⚠️ 실패: {e}", 500

# =========================
# ✅ 초기화(비동기): GET/POST/패스/쿼리 모두 허용 + 즉시 200 응답
# =========================
@app.route("/reset-all", methods=["GET","POST"])
@app.route("/reset-all/<key>", methods=["GET","POST"])
def reset_all(key=None):
    # 키 추출 (path → query → json 순)
    req_key = key or request.args.get("key") or (request.json.get("key") if request.is_json else None)
    if req_key != "3572":
        print(f"[RESET] 인증 실패 from {request.remote_addr} path={request.path}"); sys.stdout.flush()
        return "❌ 인증 실패", 403

    # 즉시 로그
    ua = request.headers.get("User-Agent", "-")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    print(f"[RESET] 요청 수신 from {ip} UA={ua}"); sys.stdout.flush()

    # 🛡️ 워치독: 초기화가 어떤 이유로든 걸려도 반드시 내려가도록 타이머 무장
    def _arm_reset_watchdog(seconds: int):
        seconds = max(30, int(seconds))
        def _kill():
            print(f"🛑 [WATCHDOG] reset watchdog fired after {seconds}s → os._exit(0)"); sys.stdout.flush()
            try:
                _release_global_lock()
            finally:
                os._exit(0)
        t = threading.Timer(seconds, _kill)
        t.daemon = True
        t.start()
        return t

    # 백그라운드 작업 정의
    def _do_reset_work():
        # ---- 환경설정(시간) 먼저 파싱하고 워치독 무장 ----
        stop_timeout = int(os.getenv("RESET_STOP_TIMEOUT", "30"))
        max_wait     = int(os.getenv("RESET_MAX_WAIT_SEC", "600"))
        poll_sec     = max(1, int(os.getenv("RESET_POLL_SEC", "3")))
        # 워치독은 전체 예상 시간보다 조금 길게(여유 60s)
        watchdog_sec = int(os.getenv("RESET_WATCHDOG_SEC", str(stop_timeout + max_wait + 60)))
        _wd = _arm_reset_watchdog(watchdog_sec)

        try:
            from data.utils import _kline_cache, _feature_cache
            import importlib

            # ===== 0) 글로벌 락 ON + 스케줄러 완전정지 =====
            _acquire_global_lock()
            _stop_all_aux_schedulers()  # 🆕 내부/정리 스케줄러 모두 정지

            # 경로 상수 로컬 바인딩
            BASE = BASE_DIR
            PERSIST = PERSIST_DIR
            LOGS = LOG_DIR
            MODELS = MODEL_DIR
            PRED_LOG = PREDICTION_LOG
            WRONG = WRONG_PREDICTIONS
            AUDIT = AUDIT_LOG
            MSG = MESSAGE_LOG
            FAIL = FAILURE_LOG

            def clear_csv(f, h):
                os.makedirs(os.path.dirname(f), exist_ok=True)
                with open(f, "w", newline="", encoding="utf-8-sig") as wf:
                    wf.write(",".join(h) + "\n")

            print("[RESET] 백그라운드 초기화 시작"); sys.stdout.flush()

            # 1) 학습 루프 정지
            try:
                if hasattr(train, "request_stop"):
                    train.request_stop()
            except Exception:
                pass

            stopped = False
            try:
                print(f"[RESET] 학습 루프 정지 시도(timeout={stop_timeout}s)"); sys.stdout.flush()
                stopped = train.stop_train_loop(timeout=stop_timeout)
            except Exception as e:
                print(f"⚠️ [RESET] stop_train_loop 예외: {e}"); sys.stdout.flush()
            print(f"[RESET] stop_train_loop 결과: {stopped}"); sys.stdout.flush()

            # 🆕 1-1) 미정지 시 폴링 대기(최대 max_wait)
            if not stopped:
                t0 = time.time()
                print(f"[RESET] 정지 대기 시작… 최대 {max_wait}s (폴링 {poll_sec}s)"); sys.stdout.flush()
                while time.time() - t0 < max_wait:
                    try:
                        if hasattr(train, "is_loop_running"):
                            running = bool(train.is_loop_running())
                            if not running:
                                stopped = True
                                break
                    except Exception:
                        pass
                    # 짧은 재시도
                    try:
                        if hasattr(train, "stop_train_loop") and train.stop_train_loop(timeout=2):
                            stopped = True
                            break
                    except Exception:
                        pass
                    time.sleep(poll_sec)
                print(f"[RESET] 정지 대기 완료 → stopped={stopped}"); sys.stdout.flush()

            # 🆕 1-2) 그래도 안 멈추면 **격리-와이프 후 하드 종료**
            if not stopped:
                print("🛑 [RESET] 루프가 종료되지 않음 → QWIPE 후 하드 종료(os._exit)"); sys.stdout.flush()
                try:
                    _quarantine_wipe_persistent()
                except Exception as e:
                    print(f"⚠️ [RESET] QWIPE 실패: {e}")
                try:
                    _release_global_lock()
                finally:
                    try:
                        _wd.cancel()
                    except Exception:
                        pass
                    os._exit(0)  # 프로세스 즉시 종료 → 플랫폼이 재기동

            # 2) 진행상태 마커 제거
            try:
                done_path = os.path.join(PERSIST, "train_done.json")
                if os.path.exists(done_path): os.remove(done_path)
            except Exception:
                pass

            # 3) 파일 정리 — 무조건 풀와이프(ssl_models 포함)
            try:
                # 대상 디렉터리 전부 제거
                for d in [MODELS, LOGS, os.path.join(PERSIST, "ssl_models")]:
                    if os.path.exists(d):
                        shutil.rmtree(d, ignore_errors=True)
                    os.makedirs(d, exist_ok=True)

                # 루트 직속 파일/잔여물 전부 제거 (락/앱필수 제외)
                keep = {os.path.basename(LOCK_DIR)}
                for name in list(os.listdir(PERSIST)):
                    p = os.path.join(PERSIST, name)
                    if name in keep:
                        continue
                    if os.path.isdir(p):
                        # 위에서 만든 기본 디렉터리( logs / models / ssl_models )는 유지
                        if name not in {"logs", "models", "ssl_models"}:
                            shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                # 의심 CSV/DB/캐시 전부 제거(안전망)
                suspect_prefixes = ("prediction_log", "eval", "message_log", "train_log",
                                    "wrong_predictions", "evaluation_audit", "failure_count",
                                    "diag", "e2e", "guan", "관우")
                for root, dirs, files in os.walk(PERSIST, topdown=False):
                    for f in files:
                        low = f.lower()
                        if low.endswith((".csv", ".db", ".json", ".txt")) or low.startswith(suspect_prefixes):
                            try: os.remove(os.path.join(root, f))
                            except Exception: pass
                    for d in dirs:
                        low = d.lower()
                        if low.startswith(suspect_prefixes) or ("관우" in d):
                            try: shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                            except Exception: pass
            except Exception as e:
                print(f"⚠️ [RESET] 풀와이프 예외: {e}"); sys.stdout.flush()

            # 4) in-memory 캐시 초기화
            try: _kline_cache.clear()
            except Exception: pass
            try: _feature_cache.clear()
            except Exception: pass

            # 5) 표준 로그 재생성(정확 헤더)
            try:
                ensure_prediction_log_exists()
                def clear_csv(f, h):
                    os.makedirs(os.path.dirname(f), exist_ok=True)
                    with open(f, "w", newline="", encoding="utf-8-sig") as wf:
                        wf.write(",".join(h) + "\n")
                clear_csv(WRONG, ["timestamp","symbol","strategy","direction","entry_price","target_price","model","predicted_class","top_k","note","success","reason","rate","return_value","label","group_id","model_symbol","model_name","source","volatility","source_exchange"])
                clear_csv(LOG_FILE, ["timestamp","symbol","strategy","model","accuracy","f1","loss","note","source_exchange","status"])
                clear_csv(AUDIT, ["timestamp","symbol","strategy","result","status"])
                clear_csv(MESSAGE_LOG, ["timestamp","symbol","strategy","message"])
                clear_csv(FAILURE_LOG, ["symbol","strategy","failures"])
            except Exception as e:
                print(f"⚠️ [RESET] 로그 재생성 예외: {e}"); sys.stdout.flush()

            # 6) diag_e2e reload
            try:
                import diag_e2e as _diag_mod
                import importlib as _imp
                _imp.reload(_diag_mod)
            except Exception:
                pass

            # 7) 메타 보정 1회
            try:
                maintenance_fix_meta.fix_all_meta_json()
            except Exception as e:
                print(f"[RESET] meta 보정 실패: {e}")

            # ✅ 정리 완료 → 락 해제 후 즉시 종료(플랫폼이 재부팅)
            print("🔚 [RESET] 정리 완료 → 프로세스 종료(os._exit)로 재부팅 진행"); sys.stdout.flush()
            _release_global_lock()
            try:
                _wd.cancel()
            except Exception:
                pass
            os._exit(0)

        except Exception as e:
            print(f"❌ [RESET] 백그라운드 초기화 예외: {e}"); sys.stdout.flush()
        finally:
            # (이중 호출이어도 안전) 혹시 못 풀었으면 풀기
            _release_global_lock()

    # 백그라운드 작업 시작 후 즉시 응답
    threading.Thread(target=_do_reset_work, daemon=True).start()
    return Response(
        "✅ 초기화 요청 접수됨. 백그라운드에서 정지→정리 후 서버 프로세스를 재시작합니다.\n"
        "로그에서 [RESET]/[SCHED]/[LOCK]/[QWIPE] 태그를 확인하세요.",
        mimetype="text/plain; charset=utf-8"
    )

# 하이픈/언더스코어 모두 허용
@app.route("/force-fix-prediction_log")
@app.route("/force-fix-prediction-log")
def force_fix_prediction_log():
    """logger의 표준 헤더로 prediction_log.csv를 안전하게 재생성"""
    try:
        from logger import ensure_prediction_log_exists
        if os.path.exists(PREDICTION_LOG):
            os.remove(PREDICTION_LOG)
        ensure_prediction_log_exists()
        print("[FORCE-FIX] prediction_log.csv 재생성 완료"); sys.stdout.flush()
        return "✅ prediction_log.csv 강제 초기화 완료"
    except Exception as e:
        return f"⚠️ 오류: {e}", 500

# ===== 로컬 개발 실행용 =====
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
    except ValueError:
        raise RuntimeError("❌ Render 환경변수 PORT가 없습니다. Render 서비스 타입 확인 필요")

    _init_background_once()
    print(f"✅ Flask 서버 실행 시작 (PORT={port})")
    app.run(host="0.0.0.0", port=port)
