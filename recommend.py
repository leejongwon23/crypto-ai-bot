# recommend.py
import datetime
from telegram_bot import send_message
from train import predict
from logger import log_prediction, evaluate_predictions
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

    for strategy in STRATEGY_GAIN_LEVELS:
        strategy_results = []  # 조건 만족 결과만 따로 모음
        all_predictions = []   # 전체 예측 결과 저장용

        for symbol in SYMBOLS:
            try:
                print(f"⏳ 예측 중: {symbol} - {strategy}")
                result = predict(symbol, strategy)
                print(f"📊 예측 결과: {result}")

                if result:
                    # 모든 예측 결과를 평가 대상으로 저장
                    log_prediction(
                        symbol=result["symbol"],
                        strategy=result["strategy"],
                        direction=result["direction"],
                        entry_price=result["price"],
                        target_price=result["target"],
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=result["confidence"]
                    )

                    min_gain = STRATEGY_GAIN_LEVELS[strategy][0]
                    if result["rate"] >= min_gain:
                        print(f"✅ 조건 만족: {symbol} - {strategy}")
                        strategy_results.append(result)
                    else:
                        print(f"❌ 수익률 미달: {result['rate']}")
                else:
                    print("❌ 예측 결과 없음")

            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

        print(f"📦 전략 [{strategy}] 조건 만족 예측 수: {len(strategy_results)}")

        if strategy_results:
            # 신뢰도 기준 Top 1 선택
            top_result = sorted(strategy_results, key=lambda x: x["confidence"], reverse=True)[0]
            print(f"📤 메시지 전송 준비: {top_result}")

            # 메시지용 log (이미 위에서 기록했지만 중복 저장해도 무방)
            msg = format_message(top_result)
            print("📨 메시지 내용:", msg)
            send_message(msg)
        else:
            print(f"⚠️ [{strategy}] 전략에 추천 조건을 만족하는 코인이 없습니다.")

if __name__ == "__main__":
    main()

    # ✅ 테스트 메시지
    test_message = "[시스템 테스트] 텔레그램 메시지가 정상 작동합니다."
    send_message(test_message)
    print("✅ 테스트 메시지 전송 완료")
