# app.py — single-source, deduped train loop (ONE concurrent loop only)

from flask import Flask, jsonify, request, Response
from recommend import main
import train, os, threading, datetime, pandas as pd, pytz, traceback, sys, shutil, csv, re
from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_test import test_all_predictions
from predict_trigger import run as trigger_run
from data.utils import SYMBOLS, get_kline_by_strategy
from visualization import generate_visual_report, generate_visuals_for_strategy
from wrong_data_loader import load_training_prediction_data
from predict import evaluate_predictions
from train import train_symbol_group_loop
import maintenance_fix_meta
from logger import ensure_prediction_log_exists
from integrity_guard import run as _integrity_check; _integrity_check()

# ✅ 종합점검 모듈(HTML/JSON + 누적 통계 지원)
from diag_e2e import run as diag_e2e_run

# ✅ cleanup 모듈 경로 보정
try:
    from scheduler_cleanup import start_cleanup_scheduler   # [KEEP]
    import safe_cleanup                                      # [KEEP]
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scheduler_cleanup import start_cleanup_scheduler    # [KEEP]
    import safe_cleanup                                      # [KEEP]

# ✅ 서버 시작 직전 용량 정리
try:
    safe_cleanup.cleanup_logs_and_models()
except Exception as e:
    print(f"[경고] startup cleanup 실패: {e}")

# ===== 경로 통일 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ← 루트 탐색용(초기화 강화에만 사용)
PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)  # ✅ 모델 디렉토리 보장

# ✅ prediction_log은 logger와 동일한 위치/헤더로 관리
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")

LOG_FILE         = os.path.join(LOG_DIR, "train_log.csv")
WRONG_PREDICTIONS= os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG        = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG      = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_LOG      = os.path.join(LOG_DIR, "failure_count.csv")

# ✅ 로그 파일 존재 보장(정확 헤더)
ensure_prediction_log_exists()

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# -----------------------------
# 학습 루프 동시 실행 방지(핵심 수정)
# -----------------------------
TRAIN_LOOP_THREAD = None
TRAIN_LOOP_LOCK = threading.Lock()

def start_train_loop_once():
    """train_symbol_group_loop를 동시 1개만 실행되게 보장"""
    global TRAIN_LOOP_THREAD
    with TRAIN_LOOP_LOCK:
        if TRAIN_LOOP_THREAD is not None and TRAIN_LOOP_THREAD.is_alive():
            print("⚠️ 학습 루프 이미 실행 중 — 재시작 생략"); sys.stdout.flush()
            return False
        TRAIN_LOOP_THREAD = threading.Thread(target=train_symbol_group_loop, daemon=True)
        TRAIN_LOOP_THREAD.start()
        print("✅ 학습 루프 스레드 시작"); sys.stdout.flush()
        return True

# -----------------------------
# 스케줄러 (평가/트리거/메타복구)
# -----------------------------
_sched = None
def start_scheduler():
    global _sched
    if _sched is not None:
        print("⚠️ 스케줄러 이미 실행 중, 재시작 생략"); sys.stdout.flush()
        return

    print(">>> 스케줄러 시작"); sys.stdout.flush()
    sched = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))

    # ✅ 전략별 평가(30분마다)
    def 평가작업(strategy):
        def wrapped():
            try:
                ts = now_kst().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[EVAL][{ts}] 전략={strategy} 시작"); sys.stdout.flush()
                evaluate_predictions(lambda sym, _: get_kline_by_strategy(sym, strategy))
            except Exception as e:
                print(f"[EVAL] {strategy} 실패: {e}")
        threading.Thread(target=wrapped, daemon=True).start()

    for strat in ["단기", "중기", "장기"]:
        sched.add_job(lambda s=strat: 평가작업(s), trigger="interval", minutes=30, id=f"eval_{strat}", replace_existing=True)

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
            from failure_db import ensure_failure_db
            print(">>> 서버 실행 준비")
            ensure_failure_db(); print("✅ failure_patterns DB 초기화 완료")

            # 학습 루프 스레드 (동시 1개 보장)
            start_train_loop_once()

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
    percent = lambda v: f"{v:.1f}%" if pd.notna(v) else "0.0%"
    logs, strategy_html, problems = {}, [], []

    file_map = {"pred": PREDICTION_LOG, "train": LOG_FILE, "audit": AUDIT_LOG, "msg": MESSAGE_LOG}
    for name, path in file_map.items():
        try:
            logs[name] = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
            if "timestamp" in logs[name].columns:
                logs[name] = logs[name][logs[name]["timestamp"].notna()]
        except Exception:
            logs[name] = pd.DataFrame()

    # 모델 파일 파싱
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    except Exception:
        model_files = []
    model_info = {}
    for f in model_files:
        m = re.match(r"(.+?)_(단기|중기|장기)_(lstm|cnn_lstm|transformer)(?:_.*)?\.pt$")
        if m:
            symbol, strat, mtype = m.groups()
            model_info.setdefault(strat, {}).setdefault(symbol, set()).add(mtype)

    for strat in ["단기", "중기", "장기"]:
        try:
            pred  = logs.get("pred",  pd.DataFrame())
            train = logs.get("train", pd.DataFrame())
            audit = logs.get("audit", pd.DataFrame())
            pred  = pred.query(f"strategy == '{strat}'")  if not pred.empty  else pd.DataFrame()
            train = train.query(f"strategy == '{strat}'") if not train.empty else pd.DataFrame()
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

            stat = lambda df, s: int(((not df.empty) and ("status" in df.columns)) and (df["status"]==s).sum()) or 0
            sn, fn, pn_, fnl = map(lambda s: stat(nvol, s), ["success", "fail", "pending", "failed"])
            sv, fv, pv, fvl = map(lambda s: stat(vol,  s), ["v_success", "v_fail", "pending", "failed"])

            def perf(df, kind="일반"):
                try:
                    s = stat(df, "v_success" if kind == "변동성" else "success")
                    f = stat(df, "v_fail"    if kind == "변동성" else "fail")
                    t = s + f
                    avg = float(df["return"].mean()) if ("return" in df) and (df.shape[0]>0) else 0.0
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

            last_train = train['timestamp'].iloc[-1] if (not train.empty and 'timestamp' in train) else '없음'
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
        symbols = request.args.get("symbols")  # ← 추가: 심볼 필터 받기

        # diag_e2e.run은 (group, view, cumulative, symbols) 시그니처 지원
        out = diag_e2e_run(group=group, view=view, cumulative=cumulative, symbols=symbols)  # ← 전달

        # diag_e2e_run이 Flask Response를 직접 반환할 수도 있음
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
        print("[RUN] 전략별 예측 실행"); sys.stdout.flush()
        for strategy in ["단기","중기","장기"]:
            main(strategy, force=True)
        return "Recommendation started"
    except Exception as e:
        traceback.print_exc(); return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    try:
        started = start_train_loop_once()
        return "✅ 전체 그룹 학습 루프 시작됨 (백그라운드)" if started else "⏳ 이미 실행 중 (재시작 생략)"
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
        group_idx = int(request.args.get("group", -1))
        if group_idx < 0 or group_idx >= len(SYMBOL_GROUPS):
            return f"❌ 잘못된 그룹 번호: {group_idx}", 400
        group_symbols = SYMBOL_GROUPS[group_idx]
        print(f"🚀 그룹 학습 요청됨 → 그룹 #{group_idx} | 심볼: {group_symbols}")
        threading.Thread(target=lambda: train.train_models(group_symbols), daemon=True).start()
        return f"✅ 그룹 #{group_idx} 학습 및 예측 시작됨"
    except Exception as e:
        traceback.print_exc(); return f"❌ 오류: {e}", 500

@app.route("/train-symbols", methods=["POST"])
def train_selected_symbols():
    try:
        symbols = request.json.get("symbols", [])
        if not isinstance(symbols, list) or not symbols:
            return "❌ 유효하지 않은 symbols 리스트", 400
        train.train_models(symbols)
        return f"✅ {len(symbols)}개 심볼 학습 시작됨"
    except Exception as e:
        return f"❌ 학습 실패: {e}", 500

@app.route("/meta-fix-now")
def meta_fix_now():
    try:
        maintenance_fix_meta.fix_all_meta_json()
        return "✅ meta.json 점검/복구 완료"
    except Exception as e:
        return f"⚠️ 실패: {e}", 500

@app.route("/reset-all")
def reset_all():
    if request.args.get("key") != "3572":
        return "❌ 인증 실패", 403
    try:
        # ==== 운영/관우 로그 완전 통일을 위한 확장 초기화 ====
        from data.utils import _kline_cache, _feature_cache
        import importlib

        def clear_csv(f, h):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "w", newline="", encoding="utf-8-sig") as wf:
                wf.write(",".join(h) + "\n")

        # 1) 진행상태 마커 제거
        done_path = os.path.join(PERSIST_DIR, "train_done.json")
        if os.path.exists(done_path): os.remove(done_path)

        # 2) 모델/로그 디렉토리 전삭제 후 재생성
        if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)

        if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
        os.makedirs(LOG_DIR, exist_ok=True)

        # 3) prediction_log.csv 포함, 루트 잔여 로그 제거
        if os.path.exists(PREDICTION_LOG):
            os.remove(PREDICTION_LOG)

        # 3-1) 작업 디렉토리/코드 루트에 남은 동일계열 파일까지 싹 정리(혼선 방지)
        suspect_names = {
            "prediction_log.csv", "wrong_predictions.csv",
            "evaluation_audit.csv", "message_log.csv",
            "failure_count.csv", "train_log.csv"
        }
        search_roots = {PERSIST_DIR, LOG_DIR, BASE_DIR, os.getcwd()}
        for root in list(search_roots):
            try:
                if not os.path.isdir(root): continue
                for fn in list(os.listdir(root)):
                    low = fn.lower()
                    if (fn in suspect_names) or low.startswith(("prediction_log", "eval", "message_log", "train_log", "wrong_predictions")):
                        try: os.remove(os.path.join(root, fn))
                        except Exception: pass
            except Exception:
                pass

        # 4) 관우/diag 추정 캐시 전부 제거 (파일/폴더)
        #    - 이름 패턴: diag*, e2e*, guan*, 관우*
        patterns = ("diag", "e2e", "guan", "관우")
        for root, dirs, files in os.walk(PERSIST_DIR, topdown=False):
            # 디렉토리 제거
            for d in list(dirs):
                name = d.lower()
                if name.startswith(patterns) or any(k in d for k in ("관우",)):
                    try: shutil.rmtree(os.path.join(root, d))
                    except Exception: pass
            # 파일 제거
            for f in list(files):
                low = f.lower()
                if low.startswith(patterns) or ("관우" in f):
                    try: os.remove(os.path.join(root, f))
                    except Exception: pass

        # 5) in-memory 캐시 초기화
        try: _kline_cache.clear()
        except Exception: pass
        try: _feature_cache.clear()
        except Exception: pass

        # 6) 표준 로그 재생성(정확한 헤더) → 운영/관우가 같은 소스만 읽도록 강제
        ensure_prediction_log_exists()
        clear_csv(WRONG_PREDICTIONS, ["timestamp","symbol","strategy","direction","entry_price","target_price","model","predicted_class","top_k","note","success","reason","rate","return_value","label","group_id","model_symbol","model_name","source","volatility","source_exchange"])
        clear_csv(LOG_FILE, ["timestamp","symbol","strategy","model","accuracy","f1","loss","note","source_exchange","status"])
        clear_csv(AUDIT_LOG, ["timestamp","symbol","strategy","result","status"])
        clear_csv(MESSAGE_LOG, ["timestamp","symbol","strategy","message"])
        clear_csv(FAILURE_LOG, ["symbol","strategy","failures"])

        # 7) diag_e2e 모듈 메모리 캐시 가능성 대비 강제 reload
        try:
            import diag_e2e as _diag_mod
            importlib.reload(_diag_mod)
        except Exception:
            pass

        # 8) 🔁 초기화 직후 학습 루프 자동 재시작 (그룹0부터, 중복 방지)
        if start_train_loop_once():
            print("✅ 초기화 이후 학습 루프 재시작됨"); sys.stdout.flush()
        else:
            print("⏳ 초기화 이후에도 학습 루프가 이미 실행 중이라 재시작 생략"); sys.stdout.flush()

        return "✅ 완전 초기화 완료"
    except Exception as e:
        return f"초기화 실패: {e}", 500

@app.route("/force-fix-prediction_log")
def force_fix_prediction_log():
    """logger의 표준 헤더로 prediction_log.csv를 안전하게 재생성"""
    try:
        from logger import ensure_prediction_log_exists
        if os.path.exists(PREDICTION_LOG):
            os.remove(PREDICTION_LOG)
        ensure_prediction_log_exists()
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
```0
