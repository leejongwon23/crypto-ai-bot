# === app.py 최종본 ===
from flask import Flask, jsonify, request
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

# ✅ cleanup 모듈 경로 보정 (src/에서 실행하든, 루트에서 실행하든 동작)
try:
    from scheduler_cleanup import start_cleanup_scheduler   # [ADD]
    import safe_cleanup                                      # [ADD]
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scheduler_cleanup import start_cleanup_scheduler    # [ADD]
    import safe_cleanup                                      # [ADD]

# ✅ 서버 시작 직전 용량 정리 (예외 가드)
try:
    safe_cleanup.cleanup_logs_and_models()
except Exception as e:
    print(f"[경고] startup cleanup 실패: {e}")

# ===== 경로 통일 =====
PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ prediction_log은 logger와 동일한 위치/헤더로 관리 (logs/아님!)
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")

LOG_FILE         = os.path.join(LOG_DIR, "train_log.csv")
WRONG_PREDICTIONS= os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG        = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG      = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_LOG      = os.path.join(LOG_DIR, "failure_count.csv")

# ✅ 로그 파일 존재 보장(정확 헤더)
ensure_prediction_log_exists()

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def start_scheduler():
    print(">>> 스케줄러 시작"); sys.stdout.flush()
    sched = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))

    # ✅ 전략별 평가(30분마다)
    def 평가작업(strategy):
        def wrapped():
            evaluate_predictions(lambda sym, _: get_kline_by_strategy(sym, strategy))
        threading.Thread(target=wrapped, daemon=True).start()

    for strat in ["단기", "중기", "장기"]:
        sched.add_job(lambda s=strat: 평가작업(s), trigger="interval", minutes=30)

    # ✅ 기타 트리거
    sched.add_job(trigger_run, "interval", minutes=30)
    sched.start()

# ===== Flask =====
app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

@app.route("/yopo-health")
def yopo_health():
    percent = lambda v: f"{v:.1f}%" if pd.notna(v) else "0.0%"
    logs, strategy_html, problems = {}, [], []

    file_map = {"pred": PREDICTION_LOG, "train": LOG_FILE, "audit": AUDIT_LOG, "msg": MESSAGE_LOG}
    for name, path in file_map.items():
        try:
            logs[name] = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
            logs[name] = logs[name][logs[name]["timestamp"].notna()] if "timestamp" in logs[name] else logs[name]
        except:
            logs[name] = pd.DataFrame()

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    model_info = {}
    for f in model_files:
        # ✅ 접미사 허용 정규식으로 보정
        m = re.match(r"(.+?)_(단기|중기|장기)_(lstm|cnn_lstm|transformer)(?:_.*)?\.pt$", f)
        if m:
            symbol, strat, mtype = m.groups()
            model_info.setdefault(strat, {}).setdefault(symbol, set()).add(mtype)

    for strat in ["단기", "중기", "장기"]:
        try:
            pred, train, audit = logs["pred"], logs["train"], logs["audit"]
            pred  = pred.query(f"strategy == '{strat}'")  if not pred.empty  else pd.DataFrame()
            train = train.query(f"strategy == '{strat}'") if not train.empty else pd.DataFrame()
            audit = audit.query(f"strategy == '{strat}'") if not audit.empty else pd.DataFrame()

            if "status" in pred.columns:
                pred["volatility"] = pred["status"].astype(str).str.startswith("v_")
            else:
                pred["volatility"] = False

            pred["return"] = pd.to_numeric(pred.get("return", pd.Series()), errors="coerce").fillna(0)
            nvol = pred[~pred["volatility"]] if not pred.empty else pd.DataFrame()
            vol  = pred[ pred["volatility"]] if not pred.empty else pd.DataFrame()

            stat = lambda df, s: len(df[df["status"] == s]) if not df.empty and "status" in df.columns else 0
            sn, fn, pn_, fnl = map(lambda s: stat(nvol, s), ["success", "fail", "pending", "failed"])
            sv, fv, pv, fvl = map(lambda s: stat(vol,  s), ["v_success", "v_fail", "pending", "failed"])

            def perf(df, kind="일반"):
                try:
                    s = stat(df, "v_success" if kind == "변동성" else "success")
                    f = stat(df, "v_fail"    if kind == "변동성" else "fail")
                    t = s + f
                    avg = df["return"].mean()
                    return {"succ": s, "fail": f, "succ_rate": s/t*100 if t else 0,
                            "fail_rate": f/t*100 if t else 0, "r_avg": avg if pd.notna(avg) else 0, "total": t}
                except:
                    return {"succ": 0, "fail": 0, "succ_rate": 0, "fail_rate": 0, "r_avg": 0, "total": 0}

            pn, pv_stats = perf(nvol, "일반"), perf(vol, "변동성")

            strat_models = model_info.get(strat, {})
            types = {"lstm": 0, "cnn_lstm": 0, "transformer": 0}
            for mtypes in strat_models.values():
                for t in mtypes: types[t] += 1
            trained_syms = [s for s, t in strat_models.items() if {"lstm","cnn_lstm","transformer"}.issubset(t)]
            untrained = sorted(set(SYMBOLS) - set(trained_syms))

            if sum(types.values()) == 0: problems.append(f"{strat}: 모델 없음")
            if sn + fn + pn_ + fnl + sv + fv + pv + fvl == 0: problems.append(f"{strat}: 예측 없음")
            if pn["succ"] + pn["fail"] == 0: problems.append(f"{strat}: 평가 미작동")
            if pn["fail_rate"]  > 50: problems.append(f"{strat}: 일반 실패율 {pn['fail_rate']:.1f}%")
            if pv_stats["fail_rate"] > 50: problems.append(f"{strat}: 변동성 실패율 {pv_stats['fail_rate']:.1f}%")

            table = "<i style='color:gray'>최근 예측 없음 또는 컬럼 부족</i>"
            required_cols = {"timestamp","symbol","direction","return","rate","status"}
            if pred.shape[0] > 0 and required_cols.issubset(set(pred.columns)):
                recent10 = pred.sort_values("timestamp").tail(10).copy()
                rows = []
                for _, r in recent10.iterrows():
                    rtn = r.get("return", 0.0) or r.get("rate", 0.0)
                    try: rtn_pct = f"{float(rtn) * 100:.2f}%"
                    except: rtn_pct = "0.00%"
                    status_icon = '✅' if r['status'] in ['success','v_success'] else '❌' if r['status'] in ['fail','v_fail'] else '⏳' if r['status']=='pending' else '🛑'
                    rows.append(f"<tr><td>{r['timestamp']}</td><td>{r['symbol']}</td><td>{r['direction']}</td><td>{rtn_pct}</td><td>{status_icon}</td></tr>")
                table = "<table border='1' style='margin-top:4px'><tr><th>시각</th><th>종목</th><th>방향</th><th>수익률</th><th>상태</th></tr>" + "".join(rows) + "</table>"

            info_html = f"""<div style='border:1px solid #aaa;margin:16px 0;padding:10px;font-family:monospace;background:#f8f8f8;'>
<b style='font-size:16px;'>📌 전략: {strat}</b><br>
- 모델 수: {sum(types.values())} (lstm={types['lstm']}, cnn={types['cnn_lstm']}, trans={types['transformer']})<br>
- 심볼 수: {len(SYMBOLS)} | 완전학습: {len(trained_syms)} | 미완성: {len(untrained)}<br>
- 최근 학습: {train['timestamp'].iloc[-1] if not train.empty else '없음'}<br>
- 최근 예측: {pred['timestamp'].iloc[-1] if not pred.empty else '없음'}<br>
- 최근 평가: {audit['timestamp'].iloc[-1] if not audit.empty else '없음'}<br>
- 예측 (일반): {sn + fn + pn_ + fnl}건 (✅{sn} ❌{fn} ⏳{pn_} 🛑{fnl})<br>
- 예측 (변동성): {sv + fv + pv + fvl}건 (✅{sv} ❌{fv} ⏳{pv} 🛑{fvl})<br>
<b style='color:#000088'>🎯 일반 예측</b>: {pn['total']}건 | {percent(pn['succ_rate'])} / {percent(pn['fail_rate'])} / {pn['r_avg']:.2f}%<br>
<b style='color:#880000'>🌪️ 변동성 예측</b>: {pv_stats['total']}건 | {percent(pv_stats['succ_rate'])} / {percent(pv_stats['fail_rate'])} / {pv_stats['r_avg']:.2f}%<br>
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
        train.train_all_models()
        return "✅ 모든 전략 학습 시작됨"
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
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
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
        df = df[df["timestamp"].notna()]
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-eval-log")
def check_eval_log():
    try:
        path = PREDICTION_LOG
        if not os.path.exists(path): return "예측 로그 없음"

        df = pd.read_csv(path, encoding="utf-8-sig")
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

@app.route("/reset-all")
def reset_all():
    if request.args.get("key") != "3572":
        return "❌ 인증 실패", 403
    try:
        from data.utils import _kline_cache, _feature_cache
        def clear_csv(f, h):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "w", newline="", encoding="utf-8-sig") as wf:
                wf.write(",".join(h) + "\n")

        done_path = os.path.join(PERSIST_DIR, "train_done.json")
        if os.path.exists(done_path): os.remove(done_path)

        if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)

        if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
        os.makedirs(LOG_DIR, exist_ok=True)

        _kline_cache.clear(); _feature_cache.clear()

        ensure_prediction_log_exists()
        clear_csv(WRONG_PREDICTIONS, ["timestamp","symbol","strategy","direction","entry_price","target_price","model","predicted_class","top_k","note","success","reason","rate","return_value","label","group_id","model_symbol","model_name","source","volatility","source_exchange"])
        clear_csv(LOG_FILE, ["timestamp","symbol","strategy","model","accuracy","f1","loss","note","source_exchange","status"])
        clear_csv(AUDIT_LOG, ["timestamp","symbol","strategy","result","status"])
        clear_csv(MESSAGE_LOG, ["timestamp","symbol","strategy","message"])
        clear_csv(FAILURE_LOG, ["symbol","strategy","failures"])
        return "✅ 완전 초기화 완료"
    except Exception as e:
        return f"초기화 실패: {e}", 500

@app.route("/force-fix-prediction-log")
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

# ===== main =====
if __name__ == "__main__":
    from failure_db import ensure_failure_db
    print(">>> 서버 실행 준비")

    ensure_failure_db(); print("✅ failure_patterns DB 초기화 완료")

    try:
        port = int(os.environ.get("PORT", 5000))
    except ValueError:
        raise RuntimeError("❌ Render 환경변수 PORT가 없습니다. Render 서비스 타입 확인 필요")

    print("🚀 첫 학습 강제 실행 시작")
    try:
        train_symbol_group_loop(); print("✅ 첫 학습 완료")
    except Exception as e:
        print(f"❌ 첫 학습 오류: {e}")

    def run_flask():
        print(f"✅ Flask 서버 실행 시작 (PORT={port})")
        app.run(host="0.0.0.0", port=port)
    threading.Thread(target=run_flask, daemon=True).start()

    def background_tasks():
        try:
            threading.Thread(target=train_symbol_group_loop, daemon=True).start()
            print("✅ 학습 루프 스레드 시작")

            # ✅ 정리 스케줄러(기본 30분) 시작 — 평가 스케줄러보다 먼저
            start_cleanup_scheduler()  # [ADD]

            try:
                start_scheduler(); print("✅ 스케줄러 시작 완료")
            except Exception as e:
                print(f"⚠️ 스케줄러 시작 실패: {e}")
            threading.Thread(target=maintenance_fix_meta.fix_all_meta_json, daemon=True).start()
            print("✅ maintenance_fix_meta 실행 완료")
            send_message("[시작] YOPO 서버 실행됨"); print("✅ Telegram 알림 발송 완료")
        except Exception as e:
            print(f"❌ 백그라운드 작업 실패: {e}")

    threading.Thread(target=background_tasks, daemon=True).start()

    import time
    while True: time.sleep(3600)
