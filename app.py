from flask import Flask
from recommend import generate_recommendation
from telegram_bot import send_recommendation
import time

app = Flask(__name__)
last_sent_time = 0  # 쿨타임 제한용 (1시간 단위)

@app.route("/run")
def run():
    global last_sent_time
    current_time = time.time()
    if current_time - last_sent_time < 3600:
        return "❗1시간 쿨타임 중입니다."

    last_sent_time = current_time

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "APTUSDT", "AVAXUSDT", "TRXUSDT", "DOGEUSDT", "LINKUSDT", "BCHUSDT"]
    results = []

    for symbol in symbols:
        r = generate_recommendation(symbol)
        if r:
            message = (
                f"\u2705 코인명: {r['symbol']}\n"
                f"\U0001F4B0 진입가: {r['entry']}\n"
                f"\u23F3 현재가: {r['current']}\n"
                f"\U0001F3AF 목표가: {r['target']} (+{r['profit_pct']}%)\n"
                f"\u26A0 손절가: {r['stop']} (-{r['loss_pct']}%)\n"
                f"\u2705 적중률: {r['hit_rate']}\n"
                f"\U0001F4CC 분석사유: {r['reason']}"
            )
            results.append(message)
            send_recommendation(message)

    return f"{len(results)}개 코인 분석 및 전송 완료"
