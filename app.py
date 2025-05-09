from flask import Flask
from recommend import main
import train
import os
import threading
from apscheduler.schedulers.background import BackgroundScheduler

# 학습 백그라운드 실행
def start_background_training():
    threading.Thread(target=train.auto_train_all, daemon=True).start()

# 예측 백그라운드 실행 (5분 간격)
def start_scheduler():
    scheduler = BackgroundScheduler()
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
        main()
        return "Recommendation started"
    except Exception as e:
        print(f"[ERROR] /run 실패: {e}")
        return f"Error: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
