from flask import Flask
from recommend import main  # 여포 1.4 메시지 포맷 포함
from train import auto_train_all  # 자동 학습 함수
import os
import threading

# ✅ 백그라운드에서 학습 실행
def start_background_training():
    threading.Thread(target=auto_train_all, daemon=True).start()

start_background_training()  # 서버 실행과 동시에 자동 학습 시작 (Render 대응)

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
    port = int(os.environ.get("PORT", 10000))  # Render 환경 대응
    app.run(host="0.0.0.0", port=port)
