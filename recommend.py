# recommend.py
# import datetime
import datetime
from telegram_bot import send_message
from train import predict
from logger import log_prediction, evaluate_predictions, get_actual_success_rate
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message  # ✅ 외부 파일에서 메시지 포맷 불러옴

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

                    # ✅ 보정된 신뢰도 계산 및 메시지 조건 적용
                    strategy_success_rate = get_actual_success_rate(result["strategy"])
                    adjusted_confidence = (result["confidence"] + strategy_success_rate) / 2

                    if adjusted_confidence >= 0.7:
                        msg = format_message(result)
                        send_message(msg)

            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
