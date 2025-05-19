# --- 필수 임포트 ---
from flask import Flask, jsonify, request, send_file
from recommend import main
import train
import os
import threading
import datetime
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import traceback
import sys
from telegram_bot import send_message
from predict_test import test_all_predictions
from data.utils import get_latest_price, SYMBOLS, get_kline_by_strategy
from predict_trigger import run as trigger_run
import shutil
import time
import csv

# --- 경로 설정 ---
PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_COUNT_LOG = os.path.join(LOG_DIR, "failure_count.csv")

# --- 전략별 조건 ---
VOLATILITY_THRESHOLD = {
    "단기": 0.003,
    "중기": 0.005,
    "장기": 0.008
}

PREDICTION_INTERVALS = {
    "단기": 3600,
    "중기": 10800,
    "장기": 21600
}
last_prediction_time = {s: 0 for s in PREDICTION_INTERVALS.keys()}

def get_symbols_by_volatility(strategy):
    if strategy not in ["단기", "중기", "장기"]:
        print(f"[SKIP] 잘못된 strategy 입력: {strategy}")
        return []
    threshold = VOLATILITY_THRESHOLD.get(strategy, 0.003)
    selected = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 20:
                continue
            vol = df["close"].pct_change().rolling(window=20).std().iloc[-1]
            if vol is not None and vol >= threshold:
                selected.append(symbol)
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return selected

def start_regular_prediction_loop():
    def loop():
        while True:
            now = time.time()
            for strategy in ["단기", "중기", "장기"]:
                interval = PREDICTION_INTERVALS[strategy]
                if now - last_prediction_time[strategy] >= interval:
                    try:
                        print(f"[정기 예측] {strategy} - {datetime.datetime.now()} - main() 실행")
                        sys.stdout.flush()
                        main(strategy)
                        last_prediction_time[strategy] = time.time()
                    except Exception as e:
                        print(f"[정기 예측 오류] {strategy}: {e}")
            time.sleep(60)
    threading.Thread(target=loop, daemon=True).start()

def start_scheduler():
    print(">>> start_scheduler() 호출됨")
    sys.stdout.flush()
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Seoul'))

    def run_evaluation():
        print(f"[평가 시작] {datetime.datetime.now()}")
        sys.stdout.flush()
        try:
            import logger
            logger.evaluate_predictions(get_latest_price)
            print("[평가 완료]")
        except Exception as e:
            print(f"[평가 오류] {e}")

    def train_short():
        print("[단기 학습 시작]")
        threading.Thread(target=train.train_model_loop, args=("단기",), daemon=True).start()

    def train_mid():
        print("[중기 학습 시작]")
        threading.Thread(target=train.train_model_loop, args=("중기",), daemon=True).start()

    def train_long():
        print("[장기 학습 시작]")
        threading.Thread(target=train.train_model_loop, args=("장기",), daemon=True).start()

    scheduler.add_job(run_evaluation, 'cron', minute=20, id='eval_loop', replace_existing=True)
    scheduler.add_job(train_short, 'cron', hour='0,3,6,9,12,15,18,21', minute=30)
    scheduler.add_job(train_mid, 'cron', hour='1,7,13,19', minute=30)
    scheduler.add_job(train_long, 'cron', hour='2,14', minute=30)
    scheduler.add_job(test_all_predictions, 'cron', minute=10, id='predict_test', replace_existing=True)
    scheduler.add_job(trigger_run, 'interval', minutes=30, id='trigger_loop', replace_existing=True)
    scheduler.start()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료")
sys.stdout.flush()

@app.route("/")
def index():
    return "Yopo server is running"

@app.route("/ping")
def ping():
    return "pong"

@app.route("/run")
def run():
    try:
        print("[RUN] main() 실행 시작")
        sys.stdout.flush()
        main()
        print("[RUN] main() 실행 완료")
        sys.stdout.flush()
        return "Recommendation started"
    except Exception as e:
        print("[ERROR] /run 실패:")
        traceback.print_exc()
        sys.stdout.flush()
        return f"Error: {e}", 500

@app.route("/reset-all")
def reset_all():
    key = request.args.get("key")
    if key != "3572":
        return "❌ 인증 실패: 잘못된 접근", 403
    try:
        def safe_clear_csv(path, headers):
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)

        safe_clear_csv(PREDICTION_LOG, ["symbol", "strategy", "direction", "price", "target", "timestamp", "confidence", "model", "success", "reason", "status"])
        safe_clear_csv(WRONG_PREDICTIONS, ["symbol", "strategy", "reason", "timestamp"])
        safe_clear_csv(LOG_FILE, ["timestamp", "symbol", "strategy", "model", "accuracy", "f1", "loss"])
        safe_clear_csv(AUDIT_LOG, ["timestamp", "symbol", "strategy", "result", "status"])
        safe_clear_csv(MESSAGE_LOG, ["timestamp", "symbol", "strategy", "message"])
        safe_clear_csv(FAILURE_COUNT_LOG, ["symbol", "strategy", "failures"])

        return "✅ 초기화 완료 (헤더 포함)"
    except Exception as e:
        return f"삭제 실패: {e}", 500

@app.route("/audit-log")
def audit_log():
    try:
        if not os.path.exists(AUDIT_LOG):
            return jsonify({"error": "audit log not found"})
        df = pd.read_csv(AUDIT_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(30).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/audit-log-download")
def audit_log_download():
    try:
        if not os.path.exists(AUDIT_LOG):
            return "평가 로그가 없습니다.", 404
        return send_file(AUDIT_LOG, mimetype="text/csv", as_attachment=True, download_name="evaluation_audit.csv")
    except Exception as e:
        return f"다운로드 실패: {e}", 500

@app.route("/health-check")
def health_check():
    results, summary = [], []
    try:
        if os.path.exists(PREDICTION_LOG):
            df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
            total = len(df)
            done = len(df[df["status"].isin(["success", "fail"])])
            results.append(f"✅ 예측 기록 OK ({total}건)")
            summary.append(f"- 평가 완료율: {(done/total*100):.1f}%" if total else "- 평가 없음")
        else:
            results.append("❌ 예측 기록 없음")
    except Exception as e:
        results.append(f"❌ 예측 확인 실패: {e}")
    try:
        if os.path.exists(MODEL_DIR):
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
        import logger
        for s in ["단기", "중기", "장기"]:
            r = logger.get_actual_success_rate(s, threshold=0.0)
            summary.append(f"- {s} 전략 성공률: {r*100:.1f}%")
    except:
        summary.append("- 전략별 성공률 확인 실패")
    formatted = "<br>".join(results + [""] + summary)
    return f"<div style='font-family:monospace; line-height:1.6;'>" + formatted + "</div>"

# --- 서버 시작 ---
if __name__ == "__main__":
    print(">>> __main__ 진입, 서버 실행 준비")
    sys.stdout.flush()
    start_scheduler()
    start_regular_prediction_loop()
    send_message("[시스템 시작] YOPO 서버가 정상적으로 실행되었으며 예측은 자동 스케줄에 따라 작동합니다.")
    print("✅ 서버 초기화 완료 (예측 루프 복원됨)")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
