import os import time from flask import Flask from recommend import recommend_all from telegram_bot import send_recommendation

✅ 모델이 없을 경우 자동 학습 실행

if not os.path.exists("best_model.pt"): import train_model  # 자동으로 모델 생성

✅ 쿨타임 설정

last_run_time = 0 COOLTIME = 60 * 60  # 1시간 쿨타임

app = Flask(name)

@app.route("/") def home(): return "🚀 Crypto AI Bot is running!"

@app.route("/run") def run(): global last_run_time now = time.time()

if now - last_run_time < COOLTIME:
    return "⏳ 쿨타임 중입니다. 잠시 후 다시 시도해주세요."

try:
    print("📊 추천 실행 시작")
    results = recommend_all()
    if results:
        for msg in results:
            send_recommendation(msg)
        last_run_time = now
        return f"✅ 총 {len(results)}개 종목 추천 완료"
    else:
        return "❌ 추천 결과 없음"
except Exception as e:
    print(f"[에러] 추천 실행 실패: {e}")
    return f"❌ 추천 실패: {e}"

if name == "main": PORT = int(os.environ.get("PORT", 10000)) app.run(host="0.0.0.0", port=PORT)

