import os
import time
from flask import Flask
from recommend import recommend_all
from telegram_bot import send_recommendation

# ✅ 모델이 없을 경우 자동 학습 실행
if not os.path.exists("best_model.pt"):
    import train_model  # 자동으로 모델 생성

# ✅ 쿨타임 설정
last_run_time = 0
COOLTIME = 60 * 60  # 1시간

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 Crypto AI Bot is running!"

@app.route("/run")
def run():
    global last_run_time
    now = time.time()

    if now - last_run_time < COOLTIME:
        return "⏳ 쿨타임 중입니다. 잠시 후 다시 시도해주세요."

    try:
        print("📊 추천 실행 시작")
        results = recommend_all()
        if results:
            for msg in results:
                send_recommendation(msg)
            last_run_time = now
            return "✅ 추천이 완료되었습니다!"
        else:
            return "❌ 추천 결과 없음 (데이터 부족 또는 분석 실패)"
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return f"❌ 분석 실패: {e}"

# ✅ Render에서 실행 가능하도록 포트 설정
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
