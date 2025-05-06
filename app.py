from flask import Flask
from recommend import generate_recommendation, fine_tune_model
from telegram_bot import send_recommendation

import datetime

app = Flask(__name__)
last_run_date = None  # ⏱️ 쿨타임: 하루 1회

@app.route("/run")
def run():
    global last_run_date
    today = datetime.date.today()
    if last_run_date == today:
        return "이미 오늘 실행됨"

    # 학습 자동화
    fine_tune_model("BTCUSDT")

    # 분석 전송
    result = generate_recommendation("BTCUSDT")
    if result:
        msg = (
            f"🔍 코인: {result['symbol']}\n"
            f"💵 진입가: {result['entry']}\n"
            f"📈 현재가: {result['current_price']}\n"
            f"🎯 목표가: {result['target']} (+{result['profit_pct']}%)\n"
            f"⚠️ 손절가: {result['stop']} (-{result['loss_pct']}%)\n"
            f"✅ 적중률: {result['hit_rate']}\n"
            f"📌 분석사유: {result['reason']}"
        )
        send_recommendation(msg)
        last_run_date = today
        return "자동학습 및 추천 전송 완료"
    return "추천 실패"
