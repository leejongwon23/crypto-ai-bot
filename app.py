from flask import Flask, jsonify
from recommend import run_recommendation
from backtest import run_backtest
from scheduler import run_schedule
import threading

app = Flask(__name__)

@app.route("/run", methods=["GET"])
def run():
    run_recommendation()
    return jsonify({"status": "recommendation triggered"})

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

def start_scheduler():
    run_schedule()  # scheduler.py 내부의 schedule.run_pending() 반복 실행

if __name__ == "__main__":
    # 스케줄러를 백그라운드에서 실행
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=5000)
