import os
import time
from flask import Flask, request
from recommend import recommend_all
from telegram_bot import send_recommendation

# ✅ 최초 실행 시 모델 학습
if not os.path.exists("best_model.pt"):
    import train_model

# ✅ 쿨타임 설정 (1시간)
last_run_time = 0
COOLTIME = 60 * 60

app = Flask(__name__)

# ✅ Render 서버 헬스 체크용 라우트 (반드시 필요)
@app.route("/healthz")
def health_check():
    return "OK", 200

# ✅ 기본 홈 페이지
@app.route("/")
def home():
    return "✅ Crypto AI Bot Server is LIVE"

# ✅ 추천 실행 엔드포인트
@app.route("/run")
def run():
    global last_run_time
    now = time.time()

    if now - last_run_time < COOLTIME:
        remain = int(COOLTIME - (now - last_run_time))
        return f"⏳ 쿨타임 중입니다. {remain}초 후 다시 시도해주세요."

    last_run_time = now

    results = recommend_all()
    if results:
        for msg in results:
            send_recommendation(msg)
        return "✅ 추천 완료"
    else:
        return "❌ 추천 결과 없음 (데이터 부족 또는 분석 실패)"

# ✅ Render에서 요구하는 포트로 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
