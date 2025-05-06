from flask import Flask
from telegram_bot import send_recommendation
from recommend import generate_recommendation

app = Flask(__name__)

# 분석 대상 고정 코인 목록 (Bybit 기준)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "BCHUSDT",
    "TRXUSDT", "SANDUSDT", "MATICUSDT", "APTUSDT", "ARBUSDT",
    "FILUSDT", "STXUSDT", "OPUSDT", "SUIUSDT", "ONDUSDT"
]

@app.route("/run")
def run():
    success_count = 0

    for symbol in SYMBOLS:
        result = generate_recommendation(symbol)
        if result:
            msg = f"""📈 코인명: {result['symbol']}
💰 진입가: {result['entry']}
🎯 목표가: {result['target']} ({result['profit_pct']}%)
⚠️ 손절가: {result['stop']} ({result['loss_pct']}%)
✅ 적중률: {result['hit_rate']}
📌 분석사유: LSTM 예측 기반 단기 {result['direction']} 확률"""
            send_recommendation(msg)
            success_count += 1

    return f"{success_count}개 코인 분석 및 전송 완료"
