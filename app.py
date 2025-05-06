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
COOLTIME = 60 * 60

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
                print(f"📨 메시지 전송됨: \n{msg}")
            last_run_time = now
            return "✅ 추천이 완료되어 텔레그램으로 전송되었습니다."
        else:
            print("❌ 추천 결과 없음")
            return "❌ 분석에 실패했습니다. 캔들 데이터 부족 또는 모델 문제일 수 있습니다."
    except Exception as e:
        print(f"❗ 서버 실행 중 오류: {str(e)}")
        return f"❗ 서버 오류 발생: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
