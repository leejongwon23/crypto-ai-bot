from flask import Flask
from recommend import main
import train
import os
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import traceback  # ← 예외 전체 로그 출력용

# 학습 백그라운드 실행
def start_background_training():
    threading.Thread(target=train.auto_train_all, daemon=True).start()

# 예측 백그라운드 실행 (5분 간격)
def start_scheduler():
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Seoul'))
    scheduler.add_job(main, 'interval', minutes=5)
    scheduler.start()

start_background_training()
start_scheduler()

app = Flask(__name__)

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
        main()
        print("[RUN] main() 실행 완료")
        return "Recommendation started"
    except Exception as e:
        print("[ERROR] /run 실패:")
        traceback.print_exc()  # ← 전체 예외 로그 출력
        return f"Error: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
