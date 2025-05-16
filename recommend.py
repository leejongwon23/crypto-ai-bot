import datetime
import os
import csv
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, evaluate_predictions, get_model_success_rate
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message

# --- 필터 설정 ---
MIN_CONFIDENCE = 0.60
MAX_TOP_CONFIDENCE = 100
MIN_GAIN_BY_STRATEGY = {
    "단기": 0.025,
    "중기": 0.05,
    "장기": 0.08
}
TOP_PER_STRATEGY = 2
FINAL_SEND_LIMIT = 10

# --- 운영 추적 로그 경로 ---
AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
os.makedirs("/persistent/logs", exist_ok=True)

def log_audit(symbol, strategy, result, status):
    now = datetime.datetime.utcnow().isoformat()
    row = {
        "timestamp": now,
        "symbol": symbol,
        "strategy": strategy,
        "result": str(result),
        "status": status
    }
    file_exists = os.path.exists(AUDIT_LOG)
    with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

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

                # ✅ 예측이 실패해도 무조건 기록
                if result is None:
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
                    log_audit(symbol, strategy, None, "예측 실패")
                    continue

                # ✅ 예측 성공 결과 기록
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
                log_audit(symbol, strategy, result, "예측+기록 완료")
                result["strategy"] = strategy
                result["symbol"] = symbol
                all_results.append(result)

            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
                log_prediction(
                    symbol=symbol,
                    strategy=strategy,
                    direction="예외",
                    entry_price=0,
                    target_price=0,
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    confidence=0.0,
                    model="unknown"
                )
                log_audit(symbol, strategy, None, f"예외 발생: {e}")

    # --- 필터 ---
    candidates = [r for r in all_results if r["confidence"] >= MIN_CONFIDENCE]
    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)[:MAX_TOP_CONFIDENCE]

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

    for r in filtered_results:
        r["success_rate"] = get_model_success_rate(r["symbol"], r["strategy"], r.get("model", "unknown"))

    final = sorted(filtered_results, key=lambda x: (-x["rate"], -x["success_rate"]))[:FINAL_SEND_LIMIT]

    if final:
        for res in final:
            msg = format_message(res)
            send_message(msg)
            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%}")
    else:
        print("⚠️ 조건 만족 결과 없음 → 메시지 전송 생략")

if __name__ == "__main__":
    main()
