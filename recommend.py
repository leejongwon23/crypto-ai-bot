import datetime
import os
from telegram_bot import send_message
from train import predict
from logger import log_prediction, evaluate_predictions
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message

def model_exists(symbol, strategy):
    model_dir = "/persistent/models"
    models = [
        f"{symbol}_{strategy}_lstm.pt",
        f"{symbol}_{strategy}_cnn_lstm.pt",
        f"{symbol}_{strategy}_transformer.pt"
    ]
    return all(os.path.exists(os.path.join(model_dir, m)) for m in models)

def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

def main():
    print("✅ 예측 평가 시작")
    evaluate_predictions(get_price_now)

    for strategy in ["단기", "중기", "장기"]:
        strategy_results = []

        for symbol in SYMBOLS:
            try:
                if not model_exists(symbol, strategy):
                    print(f"❌ 모델 없음: {symbol} - {strategy}")
                    continue

                print(f"⏳ 예측 중: {symbol} - {strategy}")
                result = predict(symbol, strategy)
                print(f"📊 예측 결과: {result}")

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

                    # ✅ 여포 3.0 필터 기준 적용
                    if (
                        result["confidence"] >= 0.7 and
                        result["rate"] >= 0.03 and
                        ("과매도" in result["reason"] or "과매수" in result["reason"])
                    ):
                        print(f"✅ 기준 만족: {symbol} - {strategy}")
                        strategy_results.append(result)
                    else:
                        print(f"❌ 필터 미통과: conf={result['confidence']}, rate={result['rate']}, reason={result['reason']}")
                else:
                    print("❌ 예측 결과 없음")
                    log_prediction(
                        symbol=symbol,
                        strategy=strategy,
                        direction="예측실패",
                        entry_price=0,
                        target_price=0,
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=0.0
                    )
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

        print(f"📦 전략 [{strategy}] 기준 통과 수: {len(strategy_results)}")

        if strategy_results:
            top_result = sorted(strategy_results, key=lambda x: x["confidence"], reverse=True)[0]
            print(f"📤 메시지 전송 준비: {top_result}")
            msg = format_message(top_result)
            print("📨 메시지 내용:", msg)
            send_message(msg)
        else:
            print(f"⚠️ [{strategy}] 추천 조건 만족 코인 없음")

if __name__ == "__main__":
    main()
    test_message = "[시스템 테스트] 텔레그램 메시지가 정상 작동합니다."
    send_message(test_message)
    print("✅ 테스트 메시지 전송 완료")
