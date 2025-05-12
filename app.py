from flask import Flask, jsonify
from recommend import main
import train
import os
import threading
import datetime
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import traceback  # 예외 전체 로그 출력용
import sys        # ← 로그 출력 강제 플러시용 추가

# ✅ logs 폴더 생성 (맨 위에서 추가됨)
os.makedirs("logs", exist_ok=True)

# 학습 백그라운드 실행
def start_background_training():
    print(">>> start_background_training() 호출됨")
    sys.stdout.flush()
    threading.Thread(target=train.auto_train_all, daemon=True).start()

# 예측 백그라운드 실행 (5분 간격)
def start_scheduler():
    print(">>> start_scheduler() 호출됨")
    sys.stdout.flush()
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Seoul'))
    scheduler.add_job(main, 'interval', minutes=5)
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

# ✅ train_log.txt 로그 출력 경로 수정됨 (logs 폴더 기준)
@app.route("/train-log")
def train_log():
    try:
        with open("logs/train_log.txt", "r") as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    except Exception as e:
        return f"로그 파일을 읽을 수 없습니다: {e}", 500

# ✅ write_test.txt 저장 테스트용 경로 추가
@app.route("/write-test")
def write_test():
    try:
        path = "write_test.txt"
        with open(path, "w") as f:
            f.write(f"[{datetime.datetime.utcnow()}] ✅ 파일 저장 테스트 성공\n")
        return f"파일 생성 성공: {path}"
    except Exception as e:
        return f"파일 생성 실패: {e}", 500

# ✅ models 폴더 내부 파일 목록 확인용 경로 추가
@app.route("/models")
def list_model_files():
    try:
        if not os.path.exists("models"):
            return "models 폴더가 존재하지 않습니다."
        files = os.listdir("models")
        if not files:
            return "models 폴더가 비어 있습니다."
        return "<pre>" + "\n".join(files) + "</pre>"
    except Exception as e:
        return f"모델 파일 확인 중 오류 발생: {e}", 500

# ✅ prediction_log.csv 최근 10줄 확인 라우트 추가
@app.route("/check-log")
def check_log():
    try:
        log_path = "prediction_log.csv"
        if not os.path.exists(log_path):
            return jsonify({"error": "prediction_log.csv not found"})

        df = pd.read_csv(log_path)
        last_10 = df.tail(10).to_dict(orient='records')
        return jsonify(last_10)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print(">>> __main__ 진입, 서버 실행 준비")
    sys.stdout.flush()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
