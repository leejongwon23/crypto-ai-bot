# app.py
import os
import time
from flask import Flask, request
from recommend import recommend_all
from telegram_bot import send_recommendation

# ✅ 모델 자동 학습 (최초 1회)
if not os.path.exists("best_model.pt"):
    import train_model  # 모델 학습 후 저장됨

# ✅ 쿨타임 설정 (1시간)
last_run_time = 0
COOLTIME = 60 * 60  # 3600초

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Crypto AI Bot Server is LIVE"

@app.route("/run")
def run():
    global last_run_time
    now = time.time()

    # 쿨타임 제한
    if now - last_run_time < COOLTIME:
        remain = int(COOLTIME - (now - last_run_time))
        return f"⏳ 쿨타임 중입니다. {remain}초 후에 다시 실행하세요."

    results = recommend_all()
    if results:
        for msg in results:
            send_recommendation(msg)
        last_run_time = now
        return f"✅ 총 {len(results)}개 종목 추천 완료 / 텔레그램 전송됨"
    else:
        return "❌ 추천할 종목이 없습니다. 데이터 부족 또는 모델 문제일 수 있습니다."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
