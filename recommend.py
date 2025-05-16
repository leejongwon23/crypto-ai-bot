import datetime
import os
import csv
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, evaluate_predictions, get_model_success_rate
from data.utils import SYMBOLS, get_realtime_prices
from src.message_formatter import format_message

# --- 필터 기준 설정 ---
MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_OVERRIDE = 0.85
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

    # --- 최상필터 구조 적용 ---
    filtered = []
    for r in all_results:
        conf = r.get("confidence", 0)
        model = r.get("model", "")
        direction = r.get("direction")
        strategy = r.get("strategy")
        rate = r.get("rate", 0)

        if conf < MIN_CONFIDENCE:
            continue

        if rate < MIN_GAIN_BY_STRATEGY.get(strategy, 0.03):
            continue

        # 모델 방향 2개 이상 일치 or confidence ≥ 0.85 우선 통과
        if model == "ensemble":
            filtered.append(r)
        elif conf >= MIN_CONFIDENCE_OVERRIDE:
            filtered.append(r)

    # 점수 계산
    for r in filtered:
        success_rate = get_model_success_rate(r["symbol"], r["strategy"], r.get("model", "unknown"))
        r["success_rate"] = success_rate
        score = success_rate * r["rate"] * r["confidence"]
        r["score"] = score

    # 점수 기준 정렬 후 Top 5 추출 (5개 미만이면 전부)
    final = sorted(filtered, key=lambda x: -x["score"])[:FINAL_SEND_LIMIT]

    if final:
        for res in final:
            msg = format_message(res)
            send_message(msg)
            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%} | 성공률: {res['success_rate']:.2%}")
    else:
        print("⚠️ 조건 만족 결과 없음 → 메시지 전송 생략")

if __name__ == "__main__":
    main()
