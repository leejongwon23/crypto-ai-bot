import datetime
from telegram_bot import send_message
from train import predict, STRATEGY_GAIN_LEVELS
from logger import log_prediction, evaluate_predictions
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message  # ✅ 외부 파일에서 메시지 포맷 불러옴

def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

def main():
    evaluate_predictions(get_price_now)
    for strategy in STRATEGY_GAIN_LEVELS:
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                if result:
                    log_prediction(
                        symbol=result["symbol"],
                        strategy=result["strategy"],
                        direction=result["direction"],
                        entry_price=result["price"],
                        target_price=result["target"],
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=result["confidence"]
                    )
                    if result["confidence"] > 0.7:
                        msg = format_message(result)
                        send_message(msg)
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
