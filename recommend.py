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
    print("✅ 예측 평가 시작")
    evaluate_predictions(get_price_now)
    all_results = []

    for strategy in STRATEGY_GAIN_LEVELS:
        for symbol in SYMBOLS:
            try:
                print(f"⏳ 예측 중: {symbol} - {strategy}")
                result = predict(symbol, strategy)
                print(f"📊 예측 결과: {result}")
                if result and result["confidence"] >= 0.85:
                    min_gain = STRATEGY_GAIN_LEVELS[strategy][0]
                    if result["rate"] >= min_gain:
                        print(f"✅ 조건 만족: {symbol} - {strategy}")
                        all_results.append(result)
                    else:
                        print(f"❌ 수익률 미달: {result['rate']}")
                else:
                    print(f"❌ 신뢰도 미달 또는 결과 없음")
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

    print(f"📦 최종 조건 만족 예측 수: {len(all_results)}")
    top_results = sorted(all_results, key=lambda x: x["confidence"], reverse=True)[:1]

    for result in top_results:
        print("📤 메시지 전송 준비:", result)
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
        print("📨 메시지 내용:", msg)
        send_message(msg)
