# --- [추천 메시지 전송 기능 전용 recommend.py] ---

import datetime
import os
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, evaluate_predictions
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message

# --- 전략별 수익률 기준 (YOPO 3.0 고정 구조) ---
STRATEGY_GAIN_LEVELS = {
    "단기": 0.03,
    "중기": 0.05,
    "장기": 0.10
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

    for strategy, min_gain in STRATEGY_GAIN_LEVELS.items():
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
                    # 예측 결과 기록 (성공/실패와 관계없이 전부 기록)
                    log_prediction(
                        symbol=result["symbol"],
                        strategy=result["strategy"],
                        direction=result["direction"],
                        entry_price=result["price"],
                        target_price=result["target"],
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=result["confidence"]
                    )

                    # 조건 만족 시 메시지 후보에 등록
                    if result["rate"] >= min_gain:
                        print(f"✅ 조건 만족: {symbol}-{strategy} (rate: {result['rate']:.2%})")
                        strategy_results.append(result)
                    else:
                        print(f"❌ 수익률 미달: {symbol}-{strategy} ({result['rate']:.2%})")

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

        # 전략별 전송 대상 결정 (최상위 1개 confidence 기준)
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
