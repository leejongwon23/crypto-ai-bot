from flask import Flask, jsonify, request
import os, sys, re, csv, shutil, threading, traceback, datetime
import pandas as pd
import pytz

from logger import ensure_prediction_log_exists
from telegram_bot import send_message
from data.utils import SYMBOLS, get_kline_by_strategy, SYMBOL_GROUPS
from predict import evaluate_predictions
from train import train_symbol_group_loop
import maintenance_fix_meta
import safe_cleanup
from scheduler_cleanup import start_cleanup_scheduler
from apscheduler.schedulers.background import BackgroundScheduler

try:
    from recommend import main
except Exception:
    main = None
try:
    import train
except Exception:
    train = None
try:
    from predict_trigger import run as trigger_run
except Exception:
    def trigger_run():
        pass

PERSIST_DIR   = "/persistent"
LOG_DIR       = os.path.join(PERSIST_DIR, "logs")
MODEL_DIR     = os.path.join(PERSIST_DIR, "models")
PREDICTION_LOG= os.path.join(PERSIST_DIR, "prediction_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

try:
    safe_cleanup.cleanup_logs_and_models()
except Exception as e:
    print(f"[경고] startup cleanup 실패: {e}")

ensure_prediction_log_exists()
print(">>> Flask 앱 생성 준비 완료"); sys.stdout.flush()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

@app.route("/")
def index():
    return "Yopo server is running"

@app.route("/ping")
def ping():
    return "pong"

@app.route("/yopo-health")
def yopo_health():
    percent = lambda v: f"{v:.1f}%" if pd.notna(v) else "0.0%"
    LOG_FILE    = os.path.join(LOG_DIR, "train_log.csv")
    AUDIT_LOG   = os.path.join(LOG_DIR, "evaluation_audit.csv")
    MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")

    logs = {}
    file_map = {"pred": PREDICTION_LOG, "train": LOG_FILE, "audit": AUDIT_LOG, "msg": MESSAGE_LOG}
    for name, path in file_map.items():
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
            logs[name] = df[df["timestamp"].notna()] if "timestamp" in df else df
        except Exception:
            logs[name] = pd.DataFrame()

    model_info = {}
    try:
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
        for f in files:
            m = re.match(r"(.+?)_(단기|중기|장기)_(lstm|cnn_lstm|transformer)(?:_.*)?\.pt$", f)
            if m:
                symbol, strat, mtype = m.groups()
                model_info.setdefault(strat, {}).setdefault(symbol, set()).add(mtype)
    except Exception:
        pass

    strategy_html, problems = [], []
    for strat in ["단기","중기","장기"]:
        try:
            pred  = logs["pred"].query(f"strategy == '{strat}'")  if not logs["pred"].empty  else pd.DataFrame()
            train = logs["train"].query(f"strategy == '{strat}'") if not logs["train"].empty else pd.DataFrame()
            audit = logs["audit"].query(f"strategy == '{strat}'") if not logs["audit"].empty else pd.DataFrame()

            if "status" in pred.columns:
                pred["volatility"] = pred["status"].astype(str).str.startswith("v_")
            else:
                pred["volatility"] = False

            pred["return"] = pd.to_numeric(pred.get("return", pd.Series()), errors="coerce").fillna(0)
            nvol = pred[~pred["volatility"]]
            vol  = pred[pred["volatility"]]

            stat = lambda df, s: len(df[df["status"] == s]) if "status" in df.columns else 0
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
                except Exception:
                    return {"succ":0,"fail":0,"succ_rate":0,"fail_rate":0,"r_avg":0,"total":0}

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

            table = "<i style='color:gray'>최근 예측 없음</i>"
            need = {"timestamp","symbol","direction","return","status"}
            if pred.shape[0] > 0 and need.issubset(set(pred.columns)):
                recent10 = pred.sort_values("timestamp").tail(10).copy()
                rows=[]
                for _, r in recent10.iterrows():
                    rtn = r.get("return", 0.0)
                    try: rtn_pct = f"{float(rtn)*100:.2f}%"
                    except: rtn_pct = "0.00%"
                    st=r.get("status","")
                    icon = '✅' if st in ['success','v_success'] else '❌' if st in ['fail','v_fail'] else '⏳' if st=='pending' else '🛑'
                    rows.append(f"<tr><td>{r.get('timestamp','')}</td><td>{r.get('symbol','')}</td><td>{r.get('direction','')}</td><td>{rtn_pct}</td><td>{icon}</td></tr>")
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
            strategy_html.append(info_html)
        except Exception as e:
            strategy_html.append(f"<div style='color:red;'>❌ {strat} 실패: {type(e).__name__} → {e}</div>")

    status = "🟢 전체 전략 정상(데이터 기준)" if not problems else "🔴 종합진단:<br>" + "<br>".join(problems)
    return f"<div style='font-family:monospace;line-height:1.6;font-size:15px;'><b>{status}</b><hr>" + "".join(strategy_html) + "</div>"

@app.route("/run")
def run():
    if main is None:
        return "recommend 모듈 없음", 500
    try:
        print("[RUN] 전략별 예측 실행"); sys.stdout.flush()
        for strategy in ["단기","중기","장기"]:
            main(strategy, force=True)
        return "Recommendation started"
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    if not train:
        return "train 모듈 없음", 500
    try:
        train.train_all_models()
        return "✅ 모든 전략 학습 시작됨"
    except Exception as e:
        return f"학습 실패: {e}", 500

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
        if not os.path.exists(PREDICTION_LOG):
            return jsonify({"error": "prediction_log.csv 없음"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        if "timestamp" in df:
            df = df[df["timestamp"].notna()]
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

_INIT_DONE = False
_INIT_LOCK = threading.Lock()

def _start_scheduler():
    print(">>> 스케줄러 시작"); sys.stdout.flush()
    sched = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))

    def _eval_job(strategy):
        def wrapped():
            try:
                evaluate_predictions(lambda sym, _: get_kline_by_strategy(sym, strategy))
            except Exception as e:
                print(f"[평가오류] {strategy}: {e}")
        threading.Thread(target=wrapped, daemon=True).start()

    for strat in ["단기","중기","장기"]:
        sched.add_job(lambda s=strat: _eval_job(s), trigger="interval", minutes=30)

    try:
        sched.add_job(trigger_run, "interval", minutes=30)
    except Exception:
        pass

    sched.start()
    print("✅ 스케줄러 시작 완료")

def _init_background_once():
    global _INIT_DONE
    with _INIT_LOCK:
        if _INIT_DONE:
            return
        print(">>> 백그라운드 초기화 시작"); sys.stdout.flush()
        try:
            start_cleanup_scheduler()
            _start_scheduler()
            threading.Thread(target=maintenance_fix_meta.fix_all_meta_json, daemon=True).start()
            if os.environ.get("DISABLE_AUTO_TRAIN","0") != "1":
                threading.Thread(target=train_symbol_group_loop, daemon=True).start()
                print("✅ 학습 루프 스레드 시작")
            try:
                send_message("[시작] YOPO 서버 실행됨")
                print("✅ Telegram 알림 발송 완료")
            except Exception as e:
                print(f"[경고] 텔레그램 알림 실패: {e}")
        except Exception as e:
            print(f"❌ 백그라운드 초기화 실패: {e}")
        _INIT_DONE = True
        print("✅ 백그라운드 초기화 완료")

@app.before_first_request
def _boot_once():
    _init_background_once()

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
    except Exception:
        port = 5000
    _init_background_once()
    print(f"✅ Flask 서버 실행 (PORT={port})")
    app.run(host="0.0.0.0", port=port)
