from flask import Flask, jsonify
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
import logger  # ✅ 통계 함수 사용을 위한 import

# ✅ Persistent 경로 기준 설정
PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_FILE = os.path.join(PERSIST_DIR, "logs", "train_log.txt")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")

os.makedirs(os.path.join(PERSIST_DIR, "logs"), exist_ok=True)

# ✅ 예측 루프 조건 함수 (09, 13, 16, 20, 22, 01시)
def is_prediction_hour():
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    return now.hour in [1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 0]

def start_background_training():
    print(">>> start_background_training() 호출됨")
    sys.stdout.flush()
    threading.Thread(target=train.auto_train_all, daemon=True).start()

def start_scheduler():
    print(">>> start_scheduler() 호출됨")
    sys.stdout.flush()
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Seoul'))

    def scheduled_job():
        if is_prediction_hour():
            print(f"[예측 루프 실행 중 - 허용된 시간대] {datetime.datetime.now()}")
            sys.stdout.flush()
            main()
        else:
            print(f"[예측 생략 - 비활성 시간대] {datetime.datetime.now()}")
            sys.stdout.flush()

    scheduler.add_job(scheduled_job, 'cron', hour='1,3,5,7,9,11,13,15,16,18,20,22,0')
    scheduler.start()

start_background_training()
start_scheduler()

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

@app.route("/train-now")
def train_now():
    try:
        print("[TRAIN-NOW] 전체 학습 즉시 실행 시작")
        sys.stdout.flush()
        threading.Thread(target=train.auto_train_all, daemon=True).start()
        return "✅ 모든 코인 + 전략 학습이 지금 바로 시작됐습니다!"
    except Exception as e:
        return f"학습 시작 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        with open(LOG_FILE, "r") as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    except Exception as e:
        return f"로그 파일을 읽을 수 없습니다: {e}", 500

@app.route("/write-test")
def write_test():
    try:
        path = os.path.join(PERSIST_DIR, "write_test.txt")
        with open(path, "w") as f:
            f.write(f"[{datetime.datetime.utcnow()}] ✅ 파일 저장 테스트 성공\n")
        return f"파일 생성 성공: {path}"
    except Exception as e:
        return f"파일 생성 실패: {e}", 500

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
        df = pd.read_csv(PREDICTION_LOG)
        last_10 = df.tail(10).to_dict(orient='records')
        return jsonify(last_10)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-wrong")
def check_wrong():
    try:
        if not os.path.exists(WRONG_PREDICTIONS):
            return jsonify({"error": "wrong_predictions.csv not found"})
        df = pd.read_csv(WRONG_PREDICTIONS)
        last_10 = df.tail(10).to_dict(orient='records')
        return jsonify(last_10)
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ 예측 정확도 통계 요약 API (예쁘게 출력)
@app.route("/check-stats")
def check_stats():
    try:
        result = logger.print_prediction_stats()
        formatted = result.replace("📊", "<b>📊</b>").replace("✅", "<b style='color:green'>✅</b>") \
                          .replace("❌", "<b style='color:red'>❌</b>").replace("⏳", "<b>⏳</b>") \
                          .replace("🎯", "<b>🎯</b>").replace("📌", "<b>📌</b>")
        formatted = formatted.replace("\n", "<br>")
        return f"<div style='font-family:monospace; line-height:1.6;'>{formatted}</div>"
    except Exception as e:
        return f"정확도 통계 출력 실패: {e}", 500

# 🔽 여기에 붙이면 됨

@app.route("/reset-all")
def reset_all():
    import glob
    try:
        # 로그 초기화
        open(PREDICTION_LOG, "w").close()
        open(WRONG_PREDICTIONS, "w").close()
        open(LOG_FILE, "w").close()

        # 모델 삭제
        for f in glob.glob(os.path.join(MODEL_DIR, "*.pt")):
            os.remove(f)

        return "✅ 예측 기록, 실패 기록, 학습 로그, 모델 전부 삭제 완료"
    except Exception as e:
        return f"삭제 실패: {e}", 500


if __name__ == "__main__":
    print(">>> __main__ 진입, 서버 실행 준비")
    sys.stdout.flush()

    test_message = "[시스템 테스트] Flask 앱이 정상적으로 실행되었으며 텔레그램 메시지도 전송됩니다."
    send_message(test_message)
    print("✅ 테스트 메시지 전송 완료")
    sys.stdout.flush()

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
