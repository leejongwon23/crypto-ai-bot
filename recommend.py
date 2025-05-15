import datetime
import os
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, evaluate_predictions, get_model_success_rate
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message

# --- 공격형 필터 설정 ---
MIN_CONFIDENCE = 0.60
MAX_TOP_CONFIDENCE = 100
MIN_GAIN_BY_STRATEGY = {
    "단기": 0.025,
    "중기": 0.05,
    "장기": 0.08
}
TOP_PER_STRATEGY = 2
FINAL_SEND_LIMIT = 10

def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

def main():
    print("✅ 예측 평가 시작")
    evaluate_predictions(get_price_now)

    all_results = []

    for strategy in ["단기", "중기", "장기"]:
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                print(f"[예측] {symbol}-{strategy} → {result}")

                if result:
                    log_prediction(
                        symbol=result["symbol"],
                        strategy=result["strategy"],
                        direction=result["direction"],
                        entry_price=result["price"],
                        target_price=result["target"],
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=result["confidence"],
                        model=result.get("model", "unknown")
                    )
                    result["strategy"] = strategy
                    result["symbol"] = symbol
                    all_results.append(result)
                else:
                    print(f"[로그 기록] {symbol}-{strategy} → 예측 실패로 기본값 기록")
                    log_prediction(
                        symbol=symbol,
                        strategy=strategy,
                        direction="예측실패",
                        entry_price=0,
                        target_price=0,
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=0.0,
                        model="unknown"
                    )

            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

    # --- 1단계: confidence 필터 ---
    candidates = [r for r in all_results if r["confidence"] >= MIN_CONFIDENCE]

    # --- 2단계: 상위 confidence 유지 ---
    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)[:MAX_TOP_CONFIDENCE]

    # --- 3단계: 전략별 최소 수익률 + Top N 추출 ---
    strategy_grouped = {}
    for r in candidates:
        strategy = r["strategy"]
        if r["rate"] < MIN_GAIN_BY_STRATEGY.get(strategy, 0.03):
            continue
        strategy_grouped.setdefault(strategy, []).append(r)

    filtered_results = []
    for strategy, items in strategy_grouped.items():
        sorted_items = sorted(items, key=lambda x: x["rate"], reverse=True)[:TOP_PER_STRATEGY]
        filtered_results.extend(sorted_items)

    # --- 4단계: success_rate 정렬 ---
    for r in filtered_results:
        r["success_rate"] = get_model_success_rate(r["symbol"], r["strategy"], r.get("model", "unknown"))

    final = sorted(filtered_results, key=lambda x: (-x["rate"], -x["success_rate"]))[:FINAL_SEND_LIMIT]

    # --- 전송 ---
    if final:
        for res in final:
            msg = format_message(res)
            send_message(msg)
            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%}")
    else:
        print("⚠️ 조건 만족 결과 없음 → 메시지 전송 생략")

if __name__ == "__main__":
    main()
