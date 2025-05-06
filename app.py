from flask import Flask
from recommend import generate_recommendation
from telegram_bot import send_recommendation
from bybit_data import get_current_price  # 추가됨
import time

app = Flask(__name__)

# 쿨타임 설정 (1시간)
last_called = 0
cooldown = 3600

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT",
    "AVAXUSDT", "ONDOUSDT", "SUIUSDT", "LINKUSDT", "DOGEUSDT",
    "TRUUSDT", "BCHUSDT", "XLMUSDT", "TRXUSDT", "HBARUSDT",
    "SANDUSDT", "BORAUSDT", "ARBUSDT", "UNIUSDT", "FILUSDT"
]

@app.route("/run")
def run():
    global last_called
    now = time.time()

    if now - last_called < cooldown:
        return "⏱ 호출 제한 중 (쿨타임 1시간 미도달)"

    last_called = now

    count = 0
    for symbol in SYMBOLS:
        current_price = get_current_price(symbol)
        result = generate_recommendation(symbol)

        if result:
            msg = f"""
📈 코인명: {result['symbol']}
💵 현재가(진입가): {current_price}
🎯 목표가: {result['target']} ({result['profit_pct']}%)
⚠️ 손절가: {result['stop']} ({result['loss_pct']}%)
✅ 적중률: {result['hit_rate']}
📌 분석사유: {result['reason']}
"""
            send_recommendation(msg.strip())
            count += 1

    return f"{count}개 코인 분석 및 전송 완료"
