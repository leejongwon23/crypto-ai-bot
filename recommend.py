# --- [추천 메시지 전송 기능 전용 recommend.py] ---
import datetime
import os
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, evaluate_predictions
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message

# --- 전략별 수익률 기준 및 추천 score 기준 ---
STRATEGY_GAIN_LEVELS = {
    "단기": {"min_rate": 0.03, "min_score": 0.60},
    "중기": {"min_rate": 0.06, "min_score": 0.65},
    "장기": {"min_rate": 0.10, "min_score": 0.70}
}

# --- 모델 파일 존재 여부 확인 ---
def model_exists(symbol, strategy):
    model_dir = "/persistent/models"
    models = [
        f"{symbol}_{strategy}_lstm.pt",
        f"{symbol}_{strategy}_cnn_lstm.pt",
        f"{symbol}_{strategy}_transformer.pt"
    ]
    return all(os.path.exists(os.path.join(model_dir, m)) for m in models)

# --- 실시간 가격 조회 함수 ---
def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

# --- 메시지 전송 메인 함수 ---
def main():
    print("✅ 예측 평가 시작")
    evaluate_predictions(get_price_now)

    for strategy, rule in STRATEGY_GAIN_LEVELS.items():
        strategy_results = []

        for symbol in SYMBOLS:
            try:
                if not model_exists(symbol, strategy):
                    print(f"❌ 모델 없음: {symbol}-{strategy} → 생략")
                    continue

                print(f"⏳ 예측 중: {symbol}-{strategy}")
                result = predict(symbol, strategy)
                print(f"📊 예측 결과: {result}")

                if result:
                    # --- 예측 결과 기록 ---
                    log_prediction(
                        symbol=result["symbol"],
                        strategy=result["strategy"],
                        direction=result["direction"],
                        entry_price=result["price"],
                        target_price=result["target"],
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=result["confidence"]
                    )

                    # --- 강화된 필터 조건: score & 수익률 ---
                    score = result["confidence"]  # 이미 가중치 반영된 avg_confidence로 처리됨
                    rate = result["rate"]

                    if score >= rule["min_score"] and rate >= rule["min_rate"]:
                        print(f"✅ 조건 만족: {symbol}-{strategy} "
                              f"(score: {score:.2f}, rate: {rate:.2%})")
                        strategy_results.append(result)
                    else:
                        print(f"❌ 조건 미달: {symbol}-{strategy} "
                              f"(score: {score:.2f}, rate: {rate:.2%})")
                else:
                    print(f"❌ 예측 결과 없음 (None)")
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
                print(f"[ERROR] {symbol}-{strategy} 예측 중 오류: {e}")

        # --- 전략별 최고 조건 결과 전송 ---
        if strategy_results:
            top = sorted(strategy_results, key=lambda x: x["confidence"], reverse=True)[0]
            print(f"📤 메시지 전송 대상: {top['symbol']} ({strategy})")
            msg = format_message(top)
            print("📨 메시지 내용:", msg)
            send_message(msg)
        else:
            print(f"⚠️ {strategy} 조건 만족 결과 없음")

# --- 수동 실행 전용 ---
if __name__ == "__main__":
    main()
