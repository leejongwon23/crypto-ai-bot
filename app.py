# YOPO 서버 진입점 - 최적화 압축 구조
from flask import Flask, jsonify, request, send_file
from recommend import main
import train, os, threading, datetime, pandas as pd, pytz, traceback, sys, shutil, csv
from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_test import test_all_predictions
from predict_trigger import run as trigger_run
from data.utils import SYMBOLS, get_kline_by_strategy

PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_LOG = os.path.join(LOG_DIR, "failure_count.csv")
os.makedirs(LOG_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def get_symbols_by_volatility(strategy):
    threshold = {"단기": 0.003, "중기": 0.005, "장기": 0.008}.get(strategy, 0.003)
    selected = []
    for sym in SYMBOLS:
        try:
            df = get_kline_by_strategy(sym, strategy)
            if df is not None and len(df) >= 20:
                vol = df["close"].pct_change().rolling(20).std().iloc[-1]
                if pd.notna(vol) and vol >= threshold:
                    selected.append(sym)
        except Exception as e:
            print(f"[ERROR] {sym}-{strategy} 변동성 계산 실패: {e}")
    return selected

def start_scheduler():
    print(">>> 스케줄러 시작"); sys.stdout.flush()
    sched = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))
    학습 = [(1,30,"단기"), (3,30,"장기"), (6,0,"중기"), (9,0,"단기"), (11,0,"중기"),
           (13,0,"장기"), (15,0,"단기"), (17,0,"중기"), (19,0,"장기"), (22,30,"단기")]
    예측 = [(7,30,s) for s in ["단기","중기","장기"]] + [(10,30,"단기"),(10,30,"중기"),
          (12,30,"중기"),(14,30,"장기"),(16,30,"단기"),(18,30,"중기")] + [(21,0,s) for s in ["단기","중기","장기"]] + [(0,0,"단기"),(0,0,"중기")]

    for h,m,s in 학습:
        sched.add_job(lambda s=s: threading.Thread(target=train.train_model_loop, args=(s,), daemon=True).start(), 'cron', hour=h, minute=m)
    for h,m,s in 예측:
        sched.add_job(lambda s=s: threading.Thread(target=main, args=(s,), daemon=True).start(), 'cron', hour=h, minute=m)

    sched.add_job(lambda: __import__('logger').evaluate_predictions(None), 'cron', minute=20)
    sched.add_job(test_all_predictions, 'cron', minute=10)
    sched.add_job(trigger_run, 'interval', minutes=30)
    sched.start()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

@app.route("/")
def index(): return "Yopo server is running"

@app.route("/ping")
def ping(): return "pong"

@app.route("/run")
def run():
    try:
        print("[RUN] main() 실행"); sys.stdout.flush()
        main(); return "Recommendation started"
    except Exception as e:
        traceback.print_exc(); return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    try:
        threading.Thread(target=train.train_all_models, daemon=True).start()
        return "✅ 모든 전략 학습 시작됨"
    except Exception as e:
        return f"학습 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        if not os.path.exists(LOG_FILE): return "학습 로그 없음"
        with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
            return "<pre>" + f.read() + "</pre>"
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

@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG): return jsonify({"error": "prediction_log.csv 없음"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-wrong")
def check_wrong():
    try:
        if not os.path.exists(WRONG_PREDICTIONS): return jsonify([])
        df = pd.read_csv(WRONG_PREDICTIONS, encoding="utf-8-sig")
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-stats")
def check_stats():
    try:
        result = __import__('logger').print_prediction_stats()
        if not isinstance(result, str): return f"형식 오류: {result}", 500
        for k,v in {"📊":"<b>📊</b>","✅":"<b style='color:green'>✅</b>","❌":"<b style='color:red'>❌</b>","⏳":"<b>⏳</b>","📌":"<b>📌</b>"}.items():
            result = result.replace(k, v)
        return f"<div style='font-family:monospace; line-height:1.6;'>{result.replace(chr(10),'<br>')}</div>"
    except Exception as e:
        return f"출력 실패: {e}", 500

@app.route("/reset-all")
def reset_all():
    if request.args.get("key") != "3572":
        return "❌ 인증 실패", 403
    try:
        def clear(f, headers):
            with open(f, "w", newline="", encoding="utf-8-sig") as x:
                csv.DictWriter(x, fieldnames=headers).writeheader()
        if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)
        clear(PREDICTION_LOG, [
            "timestamp", "symbol", "strategy", "direction", "entry_price", "target_price",
            "confidence", "model", "rate", "status", "reason", "return"  # ✅ 여기에 return 추가됨
        ])
        clear(WRONG_PREDICTIONS, ["symbol", "strategy", "reason", "timestamp"])
        clear(LOG_FILE, ["timestamp", "symbol", "strategy", "model", "accuracy", "f1", "loss"])
        clear(AUDIT_LOG, ["timestamp", "symbol", "strategy", "result", "status"])
        clear(MESSAGE_LOG, ["timestamp", "symbol", "strategy", "message"])
        clear(FAILURE_LOG, ["symbol", "strategy", "failures"])
        return "✅ 초기화 완료"
    except Exception as e:
        return f"초기화 실패: {e}", 500


@app.route("/audit-log")
def audit_log():
    try:
        if not os.path.exists(AUDIT_LOG): return jsonify({"error": "없음"})
        df = pd.read_csv(AUDIT_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(30).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/audit-log-download")
def audit_log_download():
    try:
        if not os.path.exists(AUDIT_LOG): return "없음", 404
        return send_file(AUDIT_LOG, mimetype="text/csv", as_attachment=True, download_name="evaluation_audit.csv")
    except Exception as e:
        return f"다운로드 실패: {e}", 500

@app.route("/yopo-health")
def yopo_health():
    import pandas as pd, os, datetime, pytz
    from collections import defaultdict

    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    percent = lambda v: f"{v:.1f}%" if pd.notna(v) else "0.0%"
    strat_html, warnings = [], []
    logs = {k: pd.read_csv(p, encoding="utf-8-sig") if os.path.exists(p) else pd.DataFrame()
            for k, p in {"pred": PREDICTION_LOG, "train": LOG_FILE, "audit": AUDIT_LOG, "msg": MESSAGE_LOG}.items()}

    for strat in ["단기","중기","장기"]:
        pred = logs["pred"].query(f"strategy == '{strat}'") if not logs["pred"].empty else pd.DataFrame()
        train = logs["train"].query(f"strategy == '{strat}'") if not logs["train"].empty else pd.DataFrame()
        audit = logs["audit"].query(f"strategy == '{strat}'") if not logs["audit"].empty else pd.DataFrame()
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt") and strat in f]
        r_pred = pred["timestamp"].iloc[-1] if not pred.empty else "없음"
        r_train = train["timestamp"].iloc[-1] if not train.empty else "없음"
        r_eval = audit["timestamp"].iloc[-1] if not audit.empty else "없음"
        stat = lambda df, t="": len(df[df["status"] == t])
        succ, fail, pend, failed = map(lambda s: stat(pred,s), ["success","fail","pending","failed"])
        nvol, vol = pred[~pred["symbol"].str.contains("_v", na=False)], pred[pred["symbol"].str.contains("_v", na=False)]

        def perf(df):
            s, f = stat(df,"success"), stat(df,"fail")
            total = s + f
            return {"succ": s, "fail": f, "succ_rate": s/total*100 if total else 0, "fail_rate": f/total*100 if total else 0, "r_avg": df.get("return", pd.Series()).mean() if not df.empty else 0}
        pn, pv = perf(nvol), perf(vol)

        if pn["fail_rate"] > 50: warnings.append(f"⚠️ {strat} 일반 실패율 {pn['fail_rate']:.1f}%")
        if pv["fail_rate"] > 50: warnings.append(f"⚠️ {strat} 변동성 실패율 {pv['fail_rate']:.1f}%")
        if succ+fail == 0: warnings.append(f"❌ {strat} 평가 작동 안됨")

        stat_html = f"""
        <div style='border:1px solid #aaa; margin:12px; padding:10px; font-family:monospace;'>
        <b>📌 전략: {strat}</b><br>
        - 모델 수: {len(models)}<br>
        - 최근 학습: {r_train}<br>
        - 최근 예측: {r_pred}<br>
        - 최근 평가: {r_eval}<br>
        - 예측 수: {succ+fail+pend+failed} (✅ {succ} / ❌ {fail} / ⏳ {pend} / 🛑 {failed})<br>
        <br><b>🎯 일반</b>: {percent(pn['succ_rate'])} / {percent(pn['fail_rate'])} / {pn['r_avg']:.2f}%<br>
        <b>🌪️ 변동성</b>: {percent(pv['succ_rate'])} / {percent(pv['fail_rate'])} / {pv['r_avg']:.2f}%<br>
        - 예측: {"✅" if succ+fail+pend+failed > 0 else "❌"} / 평가: {"✅" if succ+fail > 0 else "⏳"} / 학습: {"✅" if r_train != "없음" else "❌"}
        </div>
        """
        recent10 = pred.tail(10)[["timestamp","symbol","direction","return","confidence","status"]]
        rows = [f"<tr><td>{r['timestamp']}</td><td>{r['symbol']}</td><td>{r['direction']}</td><td>{r['return']:.2f}%</td><td>{r['confidence']}%</td><td>{'✅' if r['status']=='success' else '❌' if r['status']=='fail' else '⏳' if r['status']=='pending' else '🛑'}</td></tr>" for _,r in recent10.iterrows()]
        table = "<table border='1'><tr><th>시각</th><th>종목</th><th>방향</th><th>수익률</th><th>신뢰도</th><th>상태</th></tr>" + "".join(rows) + "</table>"
        strat_html.append(stat_html + f"<b>📋 {strat} 최근 예측</b><br>{table}")

    status = "🟢 정상 작동 중" if not warnings else "🔴 진단 요약:<br>" + "<br>".join(warnings)
    return f"<div style='font-family:monospace; line-height:1.6;'><b>{status}</b><hr>" + "".join(strat_html) + "</div>"

if __name__ == "__main__":
    print(">>> 서버 실행 준비"); sys.stdout.flush()
    threading.Thread(target=start_scheduler, daemon=True).start()
    threading.Thread(target=lambda: send_message("[시작] YOPO 서버 실행됨"), daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
