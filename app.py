from flask import Flask
from recommend import analyze
from telegram_bot import send_recommendation
import time

app = Flask(__name__)
last_run_time = 0
COOLTIME = 3600  # 1시간 쿨타임 (초 단위)

@app.route("/")
def home():
    return "🔄 Crypto AI Bot is live."

@app.route("/run")
def run():
    global last_run_time
    now = time.time()
    if now - last_run_time < COOLTIME:
        return f"🕒 잠시 후 다시 시도해주세요. 쿨타임 남음: {int(COOLTIME - (now - last_run_time))}초"

    last_run_time = now
    results = analyze()
    for r in results:
        message = f"""📊 [LSTM 전략 분석 결과]

📌 코인명: {r['symbol']}
💰 진입가: {r['entry']:.2f}
📈 현재가: {r['current']:.2f}
🎯 목표가: {r['target']:.2f} (+{r['profit_pct']}%)
🛑 손절가: {r['stop']:.2f} (-{r['loss_pct']}%)
📊 방향성: {"📈 상승" if r['target'] > r['entry'] else "📉 하락"}
📡 적중률: {r['hit_rate']}
📌 분석근거: {r['reason']}
"""
        send_recommendation(message)
    return "✅ 분석 완료 및 전송됨."
