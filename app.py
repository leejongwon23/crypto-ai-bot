from flask import Flask, jsonify, request
from recommend import main
import train, os, threading, datetime, pandas as pd, pytz, traceback, sys, shutil, csv
from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_test import test_all_predictions
from predict_trigger import run as trigger_run
from data.utils import SYMBOLS, get_kline_by_strategy

PERSIST_DIR = "/persistent"
LOG_DIR, MODEL_DIR = os.path.join(PERSIST_DIR, "logs"), os.path.join(PERSIST_DIR, "models")
LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_LOG = os.path.join(LOG_DIR, "failure_count.csv")
os.makedirs(LOG_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def get_symbols_by_volatility(strategy):
    threshold = {"단기":0.003,"중기":0.005,"장기":0.008}.get(strategy, 0.003)
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
    for h,m,strategy in 학습:
        sched.add_job(lambda strategy=strategy: threading.Thread(target=train.train_model_loop,args=(strategy,),daemon=True).start(),'cron',hour=h,minute=m)
    for h,m,strategy in 예측:
        sched.add_job(lambda strategy=strategy: threading.Thread(target=main,args=(strategy,),daemon=True).start(),'cron',hour=h,minute=m)
    sched.add_job(lambda: __import__('logger').evaluate_predictions(None), 'cron', minute=20)
    sched.add_job(test_all_predictions, 'cron', minute=10)
    sched.add_job(trigger_run, 'interval', minutes=30)
    sched.start()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

@app.route("/yopo-health")
def yopo_health():
    percent = lambda v: f"{v:.1f}%" if pd.notna(v) else "0.0%"
    logs, strategy_html, problems = {}, [], []
    for name, path in {"pred":PREDICTION_LOG,"train":LOG_FILE,"audit":AUDIT_LOG,"msg":MESSAGE_LOG}.items():
        try:
            logs[name] = pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else pd.DataFrame()
            if logs[name].empty or logs[name].shape[1]==0: logs[name] = pd.DataFrame()
        except Exception as e:
            print(f"[경고] 로그 로드 실패: {name} - {e}"); logs[name] = pd.DataFrame()
    for strategy in ["단기", "중기", "장기"]:
        try:
            pred = logs["pred"].query(f"strategy == '{strategy}'") if not logs["pred"].empty else pd.DataFrame()
            train = logs["train"].query(f"strategy == '{strategy}'") if not logs["train"].empty else pd.DataFrame()
            audit = logs["audit"].query(f"strategy == '{strategy}'") if not logs["audit"].empty else pd.DataFrame()
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
            models = [f for f in model_files if strategy in f]
            r_pred = pred["timestamp"].iloc[-1] if not pred.empty and "timestamp" in pred.columns else "없음"
            r_train = train["timestamp"].iloc[-1] if not train.empty and "timestamp" in train.columns else "없음"
            r_eval = audit["timestamp"].iloc[-1] if not audit.empty and "timestamp" in audit.columns else "없음"
            stat = lambda df,t="":len(df[df["status"]==t]) if not df.empty and "status" in df.columns else 0
            succ, fail, pend, failed = map(lambda s: stat(pred,s), ["success","fail","pending","failed"])
            nvol = pred[~pred["symbol"].astype(str).str.contains("_v", na=False)] if "symbol" in pred.columns else pd.DataFrame()
            vol  = pred[pred["symbol"].astype(str).str.contains("_v", na=False)] if "symbol" in pred.columns else pd.DataFrame()
            def perf(df): 
                try: s,f = stat(df,"success"),stat(df,"fail"); total=s+f
                except: return {"succ":0,"fail":0,"succ_rate":0,"fail_rate":0,"r_avg":0,"total":0}
                avg = pd.to_numeric(df.get("return", pd.Series()), errors='coerce').mean()
                return {"succ":s,"fail":f,"succ_rate":s/total*100 if total else 0,"fail_rate":f/total*100 if total else 0,"r_avg":avg if pd.notna(avg) else 0,"total":total}
            pn, pv = perf(nvol), perf(vol)
            if len(models)==0: problems.append(f"{strategy}: 모델 없음")
            if succ+fail+pend+failed==0: problems.append(f"{strategy}: 예측 없음")
            if succ+fail==0: problems.append(f"{strategy}: 평가 미작동")
            if pn["fail_rate"]>50: problems.append(f"{strategy}: 일반 실패율 {pn['fail_rate']:.1f}%")
            if pv["fail_rate"]>50: problems.append(f"{strategy}: 변동성 실패율 {pv['fail_rate']:.1f}%")
            html = f"""<div style='border:1px solid #aaa; margin:16px 0; padding:10px; font-family:monospace; background:#f8f8f8;'>
<b style='font-size:16px;'>📌 전략: {strategy}</b><br>
- 모델 수: {len(models)}<br>
- 최근 학습: {r_train}<br>
- 최근 예측: {r_pred}<br>
- 최근 평가: {r_eval}<br>
- 예측 수: {succ+fail+pend+failed} (✅ {succ} / ❌ {fail} / ⏳ {pend} / 🛑 {failed})<br><br>
<b style='color:#000088'>🎯 일반 예측</b>: 총 {pn['total']}건 | 성공률: {percent(pn['succ_rate'])} / 실패율: {percent(pn['fail_rate'])} / 평균 수익률: {pn['r_avg']:.2f}%<br>
<b style='color:#880000'>🌪️ 변동성 예측</b>: 총 {pv['total']}건 | 성공률: {percent(pv['succ_rate'])} / 실패율: {percent(pv['fail_rate'])} / 평균 수익률: {pv['r_avg']:.2f}%<br>
- 예측: {"✅" if succ+fail+pend+failed>0 else "❌"} / 평가: {"✅" if succ+fail>0 else "⏳"} / 학습: {"✅" if r_train != "없음" else "❌"}
</div>"""
            if not pred.empty and all(c in pred.columns for c in ["timestamp","symbol","direction","return","confidence","status"]):
                recent10 = pred.tail(10).copy()
                recent10["return"] = pd.to_numeric(recent10["return"], errors='coerce').fillna(0)
                recent10["confidence"] = pd.to_numeric(recent10["confidence"], errors='coerce').fillna(0)
                rows = [f"<tr><td>{r['timestamp']}</td><td>{r['symbol']}</td><td>{r['direction']}</td><td>{r['return']:.2f}%</td><td>{r['confidence']:.1f}%</td><td>{'✅' if r['status']=='success' else '❌' if r['status']=='fail' else '⏳' if r['status']=='pending' else '🛑'}</td></tr>" for _,r in recent10.iterrows()]
                table = "<table border='1' style='margin-top:4px'><tr><th>시각</th><th>종목</th><th>방향</th><th>수익률</th><th>신뢰도</th><th>상태</th></tr>" + "".join(rows) + "</table>"
            else: table = "<i>최근 예측 기록 없음</i>"
            strategy_html.append(html + f"<b>📋 {strategy} 최근 예측</b><br>{table}")
        except Exception as e:
            strategy_html.append(f"<div style='color:red;'>❌ 전략 처리 실패: {e}</div>")
    status = "🟢 전체 전략 정상 작동 중" if not problems else "🔴 종합진단 요약:<br>" + "<br>".join(problems)
    return f"<div style='font-family:monospace; line-height:1.6; font-size:15px;'><b>{status}</b><hr>" + "".join(strategy_html) + "</div>"

@app.route("/reset-all")
def reset_all():
    if request.args.get("key") != "3572": return "❌ 인증 실패", 403
    try:
        def clear(f,h): open(f,"w",newline="",encoding="utf-8-sig").write(",".join(h)+"\n")
        if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)
        clear(PREDICTION_LOG,["timestamp","symbol","strategy","direction","entry_price","target_price","confidence","model","rate","status","reason","return"])
        clear(WRONG_PREDICTIONS,["symbol","strategy","reason","timestamp"])
        clear(LOG_FILE,["timestamp","symbol","strategy","model","accuracy","f1","loss"])
        clear(AUDIT_LOG,["timestamp","symbol","strategy","result","status"])
        clear(MESSAGE_LOG,["timestamp","symbol","strategy","message"])
        clear(FAILURE_LOG,["symbol","strategy","failures"])
        return "✅ 초기화 완료"
    except Exception as e: return f"초기화 실패: {e}", 500

@app.route("/force-fix-prediction-log")
def force_fix_prediction_log():
    try:
        headers = ["timestamp","symbol","strategy","direction","entry_price","target_price","confidence","model","rate","status","reason","return"]
        with open(PREDICTION_LOG,"w",newline="",encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()
        return "✅ prediction_log.csv 강제 초기화 완료"
    except Exception as e: return f"⚠️ 오류: {e}", 500

@app.route("/")
def index(): return "Yopo server is running"

@app.route("/ping")
def ping(): return "pong"

@app.route("/run")
def run():
    try: print("[RUN] main() 실행"); sys.stdout.flush(); main(); return "Recommendation started"
    except Exception as e: traceback.print_exc(); return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    try: threading.Thread(target=train.train_all_models, daemon=True).start(); return "✅ 모든 전략 학습 시작됨"
    except Exception as e: return f"학습 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        if not os.path.exists(LOG_FILE): return "학습 로그 없음"
        df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
        if df.empty or df.shape[1]==0: return "학습 기록 없음"
        return "<pre>" + df.to_csv(index=False) + "</pre>"
    except Exception as e: return f"읽기 오류: {e}", 500

@app.route("/models")
def list_models():
    try:
        if not os.path.exists(MODEL_DIR): return "models 폴더 없음"
        files = os.listdir(MODEL_DIR)
        return "<pre>" + "\n".join(files) + "</pre>" if files else "models 폴더 비어 있음"
    except Exception as e: return f"오류: {e}", 500

@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG): return jsonify({"error": "prediction_log.csv 없음"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e: return jsonify({"error": str(e)})

if __name__ == "__main__":
    print(">>> 서버 실행 준비"); sys.stdout.flush()
    threading.Thread(target=start_scheduler, daemon=True).start()
    threading.Thread(target=lambda: send_message("[시작] YOPO 서버 실행됨"), daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
