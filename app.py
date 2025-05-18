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
import logger
from predict_test import test_all_predictions
from data.utils import get_latest_price, SYMBOLS, get_kline_by_strategy
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

VOLATILITY_THRESHOLD = {
    "단기": 0.003,
    "중기": 0.005,
    "장기": 0.008
}

def get_symbols_by_volatility(strategy):
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
    def loop(strategy, interval_sec):
        while True:
            try:
                print(f"[정기 예측] {strategy} - {datetime.datetime.now()} - main() 실행")
                sys.stdout.flush()
                main(strategy)
            except Exception as e:
                print(f"[정기 예측 오류] {strategy}: {e}")
            time.sleep(interval_sec)
    threading.Thread(target=loop, args=("단기", 7200), daemon=True).start()
    threading.Thread(target=loop, args=("중기", 21600), daemon=True).start()
    threading.Thread(target=loop, args=("장기", 86400), daemon=True).start()

def start_volatility_prediction_loop():
    def loop():
        while True:
            try:
                now = datetime.datetime.now()
                print(f"[변동성 예측] {now} - 조건 기반 전략 예측 실행")
                sys.stdout.flush()
                for strategy in ["단기", "중기", "장기"]:
                    volatile = get_symbols_by_volatility(strategy)
                    if volatile:
                        print(f"[트리거] {strategy} 변동성 높은 코인 {len(volatile)}개 → 예측 실행")
                        main(strategy)
                    else:
                        print(f"[스킵] {strategy} 변동성 기준 미달")
            except Exception as e:
                print(f"[변동성 예측 오류] {e}")
            time.sleep(3600)
    threading.Thread(target=loop, daemon=True).start()

def start_scheduler():
    print(">>> start_scheduler() 호출됨")
    sys.stdout.flush()
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Seoul'))

    def run_evaluation():
        print(f"[평가 시작] {datetime.datetime.now()}")
        sys.stdout.flush()
        try:
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
    scheduler.start()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료")
sys.stdout.flush()

# --- API 라우트 ---
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

@app.route("/train-now")
def train_now():
    try:
        print("[TRAIN-NOW] 전체 학습 즉시 실행 시작")
        sys.stdout.flush()
        threading.Thread(target=train.train_all_models, daemon=True).start()
        return "✅ 모든 코인 + 전략 학습이 지금 바로 시작됐습니다!"
    except Exception as e:
        return f"학습 시작 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        if not os.path.exists(LOG_FILE):
            return "아직 학습 로그가 없습니다."
        with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
            return "<pre>" + f.read() + "</pre>"
    except Exception as e:
        return f"로그 파일을 읽을 수 없습니다: {e}", 500

@app.route("/models")
def list_model_files():
    try:
        if not os.path.exists(MODEL_DIR):
            return "models 폴더가 존재하지 않습니다."
        files = os.listdir(MODEL_DIR)
        if not files:
            return "models 폴더가 비어 있습니다."
        return "<pre>" + "\n".join(files) + "</pre>"
    except Exception as e:
        return f"모델 파일 확인 중 오류 발생: {e}", 500

@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG):
            return jsonify({"error": "prediction_log.csv not found"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-wrong")
def check_wrong():
    try:
        if not os.path.exists(WRONG_PREDICTIONS) or os.path.getsize(WRONG_PREDICTIONS) == 0:
            return jsonify([])
        df = pd.read_csv(WRONG_PREDICTIONS, encoding="utf-8-sig")
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-stats")
def check_stats():
    try:
        result = logger.print_prediction_stats()
        if not isinstance(result, str):
            return f"출력 형식 오류: {result}", 500
        formatted = result.replace("\n", "<br>").replace("📊", "<b>📊</b>") \
                          .replace("✅", "<b style='color:green'>✅</b>") \
                          .replace("❌", "<b style='color:red'>❌</b>") \
                          .replace("⏳", "<b>⏳</b>").replace("🎯", "<b>🎯</b>") \
                          .replace("📌", "<b>📌</b>")
        return f"<div style='font-family:monospace; line-height:1.6;'>{formatted}</div>"
    except Exception as e:
        return f"정확도 통계 출력 실패: {e}", 500

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
            total, done = len(df), len(df[df["status"].isin(["success", "fail"])])
            results.append(f"✅ 예측 기록 OK ({total}건)")
            summary.append(f"- 평가 완료율: {(done/total*100):.1f}%" if total else "- 평가 없음")
        else:
            results.append("❌ 예측 기록 없음")
    except Exception as e:
        results.append(f"❌ 예측 확인 실패: {e}")
    try:
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")] if os.path.exists(MODEL_DIR) else []
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
            r = logger.get_actual_success_rate(s, threshold=0.0)
            summary.append(f"- {s} 전략 성공률: {r*100:.1f}%")
    except:
        summary.append("- 전략별 성공률 확인 실패")
    formatted = "<br>".join(results + [""] + summary)
    return f"<div style='font-family:monospace; line-height:1.6;'>{formatted}</div>"

# --- 서버 시작 ---
if __name__ == "__main__":
    print(">>> __main__ 진입, 서버 실행 준비")
    sys.stdout.flush()
    start_scheduler()
    # ✅ 자동 루프 제거 (정기 예측, 변동성 예측)
    # start_regular_prediction_loop()
    # start_volatility_prediction_loop()
    send_message("[시스템 시작] YOPO 서버가 정상적으로 실행되었으며 예측은 자동 스케줄에 따라 작동합니다.")
    print("✅ 서버 초기화 완료 (자동 주기 루프 대기 중)")
    sys.stdout.flush()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
