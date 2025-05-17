import datetime
import os
import csv
import threading
from telegram_bot import send_message
from predict import predict
from logger import (
    log_prediction, evaluate_predictions, get_model_success_rate,
    get_actual_success_rate, get_strategy_eval_count
)
from data.utils import SYMBOLS, get_realtime_prices, get_kline_by_strategy
from src.message_formatter import format_message
import train
import sys

# --- 필터 기준 ---
MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_OVERRIDE = 0.85
SUCCESS_RATE_THRESHOLD = 0.70
VOLATILITY_THRESHOLD = 0.003
FAILURE_TRIGGER_LIMIT = 3
MIN_SCORE_THRESHOLD = 0.005
STRATEGY_BAN_THRESHOLD = 0.40
FINAL_SEND_LIMIT = 5

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
    print(">>> [main] recommend.py 진입 성공")
    sys.stdout.flush()

    print("✅ 예측 평가 시작")
    sys.stdout.flush()
    try:
        evaluate_predictions(get_price_now)
    except Exception as e:
        print(f"[ERROR] 평가 루틴 실패: {e}")
        sys.stdout.flush()

    all_results = []
    try:
        failure_map = load_failure_count()
    except Exception as e:
        print(f"[ERROR] 실패 카운트 로딩 실패: {e}")
        failure_map = {}

    try:
        banned_strategies = set()
        for s in ["단기", "중기", "장기"]:
            eval_count = get_strategy_eval_count(s)
            if eval_count >= 10:
                sr = get_actual_success_rate(s, threshold=0.0)
                if sr < STRATEGY_BAN_THRESHOLD:
                    banned_strategies.add(s)
    except Exception as e:
        print(f"[ERROR] 전략 성공률 평가 실패: {e}")
        banned_strategies = set()

    for strategy in ["단기", "중기", "장기"]:
        if strategy in banned_strategies:
            print(f"[차단] 전략 성공률 낮음 → {strategy} 제외")
            continue

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
                sys.stdout.flush()

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

                log_audit(symbol, strategy, result, "예측 성공" if result.get("success") else "예측 실패")

                key = f"{symbol}-{strategy}"
                if not result.get("success", False):
                    failure_map[key] = failure_map.get(key, 0) + 1
                    if failure_map[key] >= FAILURE_TRIGGER_LIMIT:
                        print(f"[학습 트리거] {symbol}-{strategy} 실패 {failure_map[key]}회 → 학습")
                        threading.Thread(target=train.train_model, args=(symbol, strategy), daemon=True).start()
                        failure_map[key] = 0
                else:
                    failure_map[key] = 0

                if result.get("success"):
                    all_results.append(result)

            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
                sys.stdout.flush()
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
        rate = r.get("rate", 0)
        symbol = r.get("symbol")
        strategy = r.get("strategy")

        if not (model == "ensemble" or conf >= MIN_CONFIDENCE_OVERRIDE):
            continue
        if conf < MIN_CONFIDENCE:
            continue

        success_rate = get_model_success_rate(symbol, strategy, model)
        if success_rate < SUCCESS_RATE_THRESHOLD:
            continue

        penalty = 1.0 - (1.0 - success_rate) ** 2
        score = conf * rate * penalty
        if score < MIN_SCORE_THRESHOLD:
            continue

        r["success_rate"] = success_rate
        r["score"] = score
        filtered.append(r)

    final = sorted(filtered, key=lambda x: -x["score"])[:FINAL_SEND_LIMIT]

    now_hour = datetime.datetime.now().hour
    if 2 <= now_hour < 6:
        print(f"[전송 차단] 현재 {now_hour}시 → 야간 전송 제한")
        return

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

    print(">>> [main] recommend.py 실행 완료")
    sys.stdout.flush()
