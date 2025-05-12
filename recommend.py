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
    all_results = []

    for strategy in STRATEGY_GAIN_LEVELS:
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                if result and result["confidence"] >= 0.85:
                    # ✅ 단기만 수익률 5% 이상, 나머지는 기존 전략 기준 유지
                    if strategy == "단기":
                        if result["rate"] >= 0.05:
                            all_results.append(result)
                    else:
                        if result["rate"] >= STRATEGY_GAIN_LEVELS[strategy][0]:
                            all_results.append(result)
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

    # ✅ 전략 구분 없이 전체 예측 중 신뢰도 Top 1개만 전송
    top_results = sorted(all_results, key=lambda x: x["confidence"], reverse=True)[:1]
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

