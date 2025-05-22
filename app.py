from flask import Flask, jsonify, request, send_file
from recommend import main
import train, os, threading, datetime, pandas as pd, pytz, traceback, sys, shutil, time, csv
from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_test import test_all_predictions
from predict_trigger import run as trigger_run
from data.utils import SYMBOLS, get_kline_by_strategy
from src.healthcheck_yopo import generate_health_report

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_COUNT_LOG = os.path.join(LOG_DIR, "failure_count.csv")
os.makedirs(LOG_DIR, exist_ok=True)

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def get_symbols_by_volatility(strategy):
    threshold_map = {"단기": 0.003, "중기": 0.005, "장기": 0.008}
    threshold = threshold_map.get(strategy, 0.003)
    selected = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 20: continue
            vol = df["close"].pct_change().rolling(window=20).std().iloc[-1]
            if vol and vol >= threshold:
                selected.append(symbol)
        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 변동성 계산 실패: {e}")
    return selected

def start_scheduler():
    print(">>> start_scheduler() 호출됨"); sys.stdout.flush()
    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))

    # 🎓 학습 스케줄 (예측보다 30~60분 앞서 실행)
    for h, m, s in [
        (7, 0, "단기"), (9, 30, "단기"), (12, 0, "단기"), (14, 30, "단기"), (17, 0, "단기"),
        (7, 0, "중기"), (15, 0, "중기"), (21, 0, "중기"),
        (6, 0, "장기")
    ]:
        scheduler.add_job(lambda s=s: threading.Thread(target=train.train_model_loop, args=(s,), daemon=True).start(),
                          'cron', hour=h, minute=m)

    # 📈 예측 스케줄
    for h, m, s in [
        (7, 30, "단기"), (10, 0, "단기"), (12, 30, "단기"), (15, 0, "단기"), (17, 30, "단기"),
        (8, 0, "중기"), (16, 0, "중기"), (22, 0, "중기"),
        (7, 0, "장기")
    ]:
        scheduler.add_job(lambda s=s: threading.Thread(target=main, args=(s,), daemon=True).start(),
                          'cron', hour=h, minute=m)

    # ✅ 기타 루틴
    scheduler.add_job(lambda: __import__('logger').evaluate_predictions(None), 'cron', minute=20)
    scheduler.add_job(test_all_predictions, 'cron', minute=10)
    scheduler.add_job(trigger_run, 'interval', minutes=30)

    scheduler.start()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

@app.route("/")
def index(): return "Yopo server is running"

@app.route("/ping")
def ping(): return "pong"

@app.route("/run")
def run():
    try:
        print("[RUN] main() 실행 시작"); sys.stdout.flush()
        main()
        print("[RUN] main() 실행 완료"); sys.stdout.flush()
        return "Recommendation started"
    except Exception as e:
        print("[ERROR] /run 실패:"); traceback.print_exc(); sys.stdout.flush()
        return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    try:
        threading.Thread(target=train.train_all_models, daemon=True).start()
        return "✅ 모든 코인 + 전략 학습이 지금 바로 시작됐습니다!"
    except Exception as e:
        return f"학습 시작 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        if not os.path.exists(LOG_FILE): return "아직 학습 로그가 없습니다."
        with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
            return "<pre>" + f.read() + "</pre>"
    except Exception as e:
        return f"로그 파일을 읽을 수 없습니다: {e}", 500

@app.route("/models")
def list_model_files():
    try:
        if not os.path.exists(MODEL_DIR): return "models 폴더가 존재하지 않습니다."
        files = os.listdir(MODEL_DIR)
        return "<pre>" + "\n".join(files) + "</pre>" if files else "models 폴더가 비어 있습니다."
    except Exception as e:
        return f"모델 파일 확인 중 오류 발생: {e}", 500

@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG): return jsonify({"error": "prediction_log.csv not found"})
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
        if not isinstance(result, str): return f"출력 형식 오류: {result}", 500
        for s, r in {"📊":"<b>📊</b>", "✅":"<b style='color:green'>✅</b>", "❌":"<b style='color:red'>❌</b>", "⏳":"<b>⏳</b>", "🎯":"<b>🎯</b>", "📌":"<b>📌</b>"}.items():
            result = result.replace(s, r)
        return f"<div style='font-family:monospace; line-height:1.6;'>" + result.replace(chr(10), "<br>") + "</div>"
    except Exception as e:
        return f"정확도 통계 출력 실패: {e}", 500

@app.route("/reset-all")
def reset_all():
    if request.args.get("key") != "3572": return "❌ 인증 실패: 잘못된 접근", 403
    try:
        def clear(path, headers):
            with open(path, "w", newline="", encoding="utf-8-sig") as f: csv.DictWriter(f, fieldnames=headers).writeheader()
        if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)
        clear(PREDICTION_LOG, ["symbol", "strategy", "direction", "price", "target", "timestamp", "confidence", "model", "success", "reason", "status"])
        clear(WRONG_PREDICTIONS, ["symbol", "strategy", "reason", "timestamp"])
        clear(LOG_FILE, ["timestamp", "symbol", "strategy", "model", "accuracy", "f1", "loss"])
        clear(AUDIT_LOG, ["timestamp", "symbol", "strategy", "result", "status"])
        clear(MESSAGE_LOG, ["timestamp", "symbol", "strategy", "message"])
        clear(FAILURE_COUNT_LOG, ["symbol", "strategy", "failures"])
        return "✅ 초기화 완료 (헤더 포함)"
    except Exception as e:
        return f"삭제 실패: {e}", 500

@app.route("/audit-log")
def audit_log():
    try:
        if not os.path.exists(AUDIT_LOG): return jsonify({"error": "audit log not found"})
        df = pd.read_csv(AUDIT_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(30).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/audit-log-download")
def audit_log_download():
    try:
        if not os.path.exists(AUDIT_LOG): return "평가 로그가 없습니다.", 404
        return send_file(AUDIT_LOG, mimetype="text/csv", as_attachment=True, download_name="evaluation_audit.csv")
    except Exception as e:
        return f"다운로드 실패: {e}", 500

@app.route("/yopo-health")
def yopo_health():
    try:
        result = generate_health_report()
        return f"<pre style='font-family:monospace; line-height:1.6'>{result}</pre>"
    except Exception as e:
        return f"[오류] 헬스체크 실패: {e}", 500

@app.route("/health-check")
def health_check():
    results, summary = [], []
    try:
        if os.path.exists(PREDICTION_LOG):
            df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
            total, done = len(df), len(df[df["status"].isin(["success", "fail"])])
            results.append(f"✅ 예측 기록 OK ({total}건)")
            summary.append(f"- 평가 완료율: {(done/total*100):.1f}%" if total else "- 평가 없음")
        else:
            results.append("❌ 예측 기록 없음")
    except Exception as e:
        results.append(f"❌ 예측 확인 실패: {e}")
    try:
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
        results.append(f"✅ 모델 파일 OK ({len(models)}개)" if models else "❌ 모델 없음")
    except Exception as e:
        results.append(f"❌ 모델 확인 실패: {e}")
    try:
        if os.path.exists(MESSAGE_LOG):
            df = pd.read_csv(MESSAGE_LOG, encoding="utf-8-sig")
            results.append(f"✅ 메시지 로그 OK ({len(df)}건)")
    except Exception as e:
        results.append(f"❌ 메시지 확인 실패: {e}")
    try:
        for s in ["단기", "중기", "장기"]:
            r = __import__('logger').get_actual_success_rate(s, threshold=0.0)
            summary.append(f"- {s} 전략 성공률: {r*100:.1f}%")
    except:
        summary.append("- 전략별 성공률 확인 실패")
    return f"<div style='font-family:monospace; line-height:1.6;'>" + "<br> ".join(results + [""] + summary) + "</div>"

if __name__ == "__main__":
    print(">>> __main__ 진입, 서버 실행 준비"); sys.stdout.flush()
    threading.Thread(target=start_scheduler, daemon=True).start()
    threading.Thread(target=lambda: send_message("[시스템 시작] YOPO 서버가 정상적으로 실행되었습니다."), daemon=True).start()
    print("✅ 서버 초기화 완료 (정기 예측 루프 포함)"); sys.stdout.flush()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
