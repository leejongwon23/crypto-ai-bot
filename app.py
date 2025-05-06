from flask import Flask
from recommend import generate_recommendation
from telegram_bot import send_recommendation

app = Flask(__name__)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "AVAXUSDT", "ONDOUSDT",
    "SUIUSDT", "LINKUSDT", "DOGEUSDT", "TRUMPUSDT", "BCHUSDT", "XLMUSDT", "TRXUSDT",
    "HBARUSDT", "SANDUSDT", "BORAUSDT", "ARBUSDT", "UNIUSDT", "FILUSDT", "APTUSDT"
]

@app.route('/')
def index():
    return "Crypto LSTM Recommendation API"

@app.route('/run')
def run():
    sent_count = 0
    for symbol in SYMBOLS:
        result = generate_recommendation(symbol)
        if not result:
            continue
        
        msg = (
            f"📈 코인명: {result['symbol']}\n"
            f"💰 진입가: {result['entry']}\n"
            f"🎯 목표가: {result['target']} (+{result['profit_pct']}%)\n"
            f"⚠️ 손절가: {result['stop']} (-{result['loss_pct']}%)\n"
            f"✅ 적중률: {result['hit_rate']}\n"
            f"📌 분석사유: {result['reason']}"
        )
        send_recommendation(msg)
        sent_count += 1

    return f"{sent_count}개 코인 분석 및 전송 완료"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
