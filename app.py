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
COOLTIME = 60 * 60  # 3600초 = 1시간

app = Flask(__name__)

# ✅ 서버 상태 확인용 루트 라우터
@app.route("/")
def home():
    return "✅ Crypto AI Bot Server is LIVE"

# ✅ Render Health Check 대응용
@app.route("/healthz")
def health():
    return "ok"

# ✅ 추천 실행 라우트
@app.route("/run")
def run():
    global last_run_time
    now = time.time()

    # ⏳ 쿨타임 제한
    if now - last_run_time < COOLTIME:
        remain = int(COOLTIME - (now - last_run_time))
        return f"⏳ 쿨타임 중입니다. {remain}초 후 다시 시도해주세요."

    last_run_time = now
    print("📊 추천 실행 시작")

    results = recommend_all()
    if results:
        for msg in results:
            send_recommendation(msg)
        print("✅ 추천 결과 전송 완료")
        return "✅ 추천 완료"
    else:
        print("❌ 추천 결과 없음 (데이터 부족 또는 분석 실패)")
        return "❌ 추천 결과 없음 (데이터 부족 또는 분석 실패)"

# ✅ Render용 포트 바인딩
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
