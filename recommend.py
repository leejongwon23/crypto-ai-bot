import datetime
import os
import csv
import threading
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, evaluate_predictions, get_model_success_rate
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message
import train  # ✅ 모델 자동 학습을 위해 추가

# --- 필터 기준 설정 ---
MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_OVERRIDE = 0.85
SUCCESS_RATE_THRESHOLD = 0.70
MIN_GAIN_BY_STRATEGY = {
    "단기": 0.03,
    "중기": 0.06,
    "장기": 0.10
}
FINAL_SEND_LIMIT = 5  # 최종 메시지 최대 개수

# --- 로그 경로 설정 ---
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

                # ✅ 모델 없음 시 자동 학습 트리거
                if result.get("reason") == "모델 없음":
                    print(f"[자동학습] {symbol}-{strategy} → 모델 없음 → 학습 시도")
                    threading.Thread(target=train.train_model, args=(symbol, strategy), daemon=True).start()

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
                except Exception as le:
                    print(f"[치명적] log_prediction 실패: {le}")
                try:
                    log_audit(symbol, strategy, None, f"예외 발생: {e}")
                except Exception as la:
                    print(f"[치명적] log_audit 실패: {la}")

    # --- 최상필터 적용 ---
    filtered = []
    for r in all_results:
        conf = r.get("confidence", 0)
        model = r.get("model", "")
        strategy = r.get("strategy")
        rate = r.get("rate", 0)

        # 필터 3. 전략별 최소 수익률
        if rate < MIN_GAIN_BY_STRATEGY.get(strategy, 0.03):
            continue

        # 필터 1. 모델 방향 일치 (ensemble이거나, conf ≥ 0.85)
        if model == "ensemble" or conf >= MIN_CONFIDENCE_OVERRIDE:
            pass
        else:
            continue

        # 필터 2. confidence ≥ 0.70
        if conf < MIN_CONFIDENCE:
            continue

        # 필터 4. success rate ≥ 70%
        success_rate = get_model_success_rate(r["symbol"], strategy, model)
        if success_rate < SUCCESS_RATE_THRESHOLD:
            continue

        r["success_rate"] = success_rate
        r["score"] = conf * rate * success_rate
        filtered.append(r)

    # 필터 5. 전략별 Top 1만 추출
    top_per_strategy = {}
    for item in filtered:
        strat = item["strategy"]
        if strat not in top_per_strategy or item["score"] > top_per_strategy[strat]["score"]:
            top_per_strategy[strat] = item

    final = list(top_per_strategy.values())
    final = sorted(final, key=lambda x: -x["score"])[:FINAL_SEND_LIMIT]

    if final:
        for res in final:
            msg = format_message(res)
            send_message(msg)
            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%} | 성공률: {res['success_rate']:.2%}")
    else:
        print("⚠️ 조건 만족 결과 없음 → 메시지 전송 생략")

if __name__ == "__main__":
    main()
