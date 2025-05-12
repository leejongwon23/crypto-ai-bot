# recommend.py
import datetime
from telegram_bot import send_message
from train import predict
from logger import log_prediction, evaluate_predictions, get_actual_success_rate
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message

# ✅ 전략별 수익률 구간 설정 (3~50%, 5~80%, 10~100%)
STRATEGY_GAIN_LEVELS = {
    "단기": [0.03, 0.50],
    "중기": [0.05, 0.80],
    "장기": [0.10, 1.00]
}

def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

def main():
    evaluate_predictions(get_price_now)
    for strategy in STRATEGY_GAIN_LEVELS:
        results = []
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                if result and result["confidence"] >= 0.8 and result["rate"] >= STRATEGY_GAIN_LEVELS[strategy][0]:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

        # ✅ 신뢰도 기준 상위 2개만 메시지 전송
        top_results = sorted(results, key=lambda x: x["confidence"], reverse=True)[:2]
        for result in top_results:
            log_prediction(
                symbol=result["symbol"],
                strategy=result["strategy"],
                direction=result["direction"],
                entry_price=result["price"],
                target_price=result["target"],
                timestamp=datetime.datetime.utcnow().isoformat(),
                confidence=result["confidence"]
            )
            msg = format_message(result)
            send_message(msg)
