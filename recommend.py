import datetime
import os
import csv
import threading
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, evaluate_predictions, get_model_success_rate, get_min_gain
from data.utils import SYMBOLS, get_realtime_prices, get_kline_by_strategy
from src.message_formatter import format_message
import train

# --- 필터 기준 설정 ---
MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_OVERRIDE = 0.85
SUCCESS_RATE_THRESHOLD = 0.70
VOLATILITY_THRESHOLD = 0.003
FINAL_SEND_LIMIT = 5
FAILURE_TRIGGER_LIMIT = 3

AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
os.makedirs("/persistent/logs", exist_ok=True)

def load_failure_count():
    if not os.path.exists(FAILURE_LOG):
        return {}
    with open(FAILURE_LOG, "r", encoding="utf-8-sig") as f:
        return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}

def save_failure_count(failure_map):
    with open(FAILURE_LOG, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "strategy", "failures"])
        writer.writeheader()
        for key, count in failure_map.items():
            symbol, strategy = key.split("-")
            writer.writerow({"symbol": symbol, "strategy": strategy, "failures": count})

def log_audit(symbol, strategy, result, status):
    now = datetime.datetime.utcnow().isoformat()
    row = {
        "timestamp": now,
        "symbol": symbol,
        "strategy": strategy,
        "result": str(result),
        "status": status
    }
    write_header = not os.path.exists(AUDIT_LOG)
    with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

def main():
    print("✅ 예측 평가 시작")
    evaluate_predictions(get_price_now)

    all_results = []
    failure_map = load_failure_count()

    for strategy in ["단기", "중기", "장기"]:
        for symbol in SYMBOLS:
            try:
                df = get_kline_by_strategy(symbol, strategy)
                if df is None or len(df) < 20:
                    print(f"[스킵] {symbol}-{strategy} → 데이터 부족")
                    continue
                volatility = df["close"].pct_change().rolling(window=20).std().iloc[-1]
                if volatility < VOLATILITY_THRESHOLD:
                    print(f"[스킵] {symbol}-{strategy} → 변동성 {volatility:.4f} < {VOLATILITY_THRESHOLD}")
                    continue

                result = predict(symbol, strategy)
                print(f"[예측] {symbol}-{strategy} → {result}")

                if not isinstance(result, dict):
                    raise ValueError("predict() 반환값이 dict가 아님")

                log_prediction(
                    symbol=result.get("symbol", symbol),
                    strategy=result.get("strategy", strategy),
                    direction=result.get("direction", "예측실패"),
                    entry_price=result.get("price", 0),
                    target_price=result.get("target", 0),
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    confidence=result.get("confidence", 0.0),
                    model=result.get("model", "unknown"),
                    success=result.get("success", False),
                    reason=result.get("reason", "예측 실패")
                )

                status_msg = "예측 성공" if result.get("success") else "예측 실패"
                log_audit(symbol, strategy, result, status_msg)

                key = f"{symbol}-{strategy}"
                if not result.get("success", False):
                    failure_map[key] = failure_map.get(key, 0) + 1
                    if failure_map[key] >= FAILURE_TRIGGER_LIMIT:
                        print(f"[학습 트리거] {symbol}-{strategy} → 실패 {failure_map[key]}회 → 학습 실행")
                        threading.Thread(target=train.train_model, args=(symbol, strategy), daemon=True).start()
                        failure_map[key] = 0
                else:
                    failure_map[key] = 0

                if result.get("success"):
                    all_results.append(result)

            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
                try:
                    log_prediction(
                        symbol=symbol,
                        strategy=strategy,
                        direction="예외",
                        entry_price=0,
                        target_price=0,
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=0.0,
                        model="unknown",
                        success=False,
                        reason=f"예외 발생: {e}"
                    )
                    log_audit(symbol, strategy, None, f"예외 발생: {e}")
                except Exception as err:
                    print(f"[치명적] 로그 기록 실패: {err}")

    save_failure_count(failure_map)

    filtered = []
    for r in all_results:
        conf = r.get("confidence", 0)
        model = r.get("model", "")
        strategy = r.get("strategy")
        symbol = r.get("symbol")
        rate = r.get("rate", 0)

        min_gain = get_min_gain(symbol, strategy)
        if rate < min_gain:
            continue
        if not (model == "ensemble" or conf >= MIN_CONFIDENCE_OVERRIDE):
            continue
        if conf < MIN_CONFIDENCE:
            continue

        success_rate = get_model_success_rate(symbol, strategy, model)
        if success_rate < SUCCESS_RATE_THRESHOLD:
            continue

        # ✅ soft penalty 적용
        penalty = 1.0 - (1.0 - success_rate) ** 2
        r["success_rate"] = success_rate
        r["score"] = conf * rate * penalty
        filtered.append(r)

    top_per_strategy = {}
    for item in filtered:
        strat = item["strategy"]
        if strat not in top_per_strategy or item["score"] > top_per_strategy[strat]["score"]:
            top_per_strategy[strat] = item

    final = list(top_per_strategy.values())
    final = sorted(final, key=lambda x: -x["score"])[:FINAL_SEND_LIMIT]

    if final:
        for res in final:
            try:
                msg = format_message(res)
                send_message(msg)
                print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%} | 성공률: {res['success_rate']:.2f}")
            except Exception as e:
                print(f"[ERROR] 메시지 전송 실패: {e}")
    else:
        print("⚠️ 조건 만족 결과 없음 → 메시지 전송 생략")

if __name__ == "__main__":
    main()
