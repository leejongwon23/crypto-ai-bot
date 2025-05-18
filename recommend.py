# --- [필수 import] ---
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
import time

# --- 필터 기준 ---
MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_OVERRIDE = 0.85
SUCCESS_RATE_THRESHOLD = 0.70
VOLATILITY_THRESHOLD = 0.003
FAILURE_TRIGGER_LIMIT = 3
MIN_SCORE_THRESHOLD = 0.005
FINAL_SEND_LIMIT = 5

AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
MESSAGE_LOG = "/persistent/logs/message_log.csv"
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

def get_symbols_by_volatility(strategy, threshold=VOLATILITY_THRESHOLD):
    selected = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 20:
                continue
            vol = df["close"].pct_change().rolling(window=20).std().iloc[-1]
            if vol is not None and vol >= threshold:
                selected.append(symbol)
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return selected

def should_predict(symbol, strategy):
    try:
        rate = get_model_success_rate(symbol, strategy, "ensemble")
        eval_count = get_strategy_eval_count(strategy)
        if eval_count < 10:
            return True
        if rate < 0.5:
            return True
        if rate > 0.85:
            return False
        return True
    except:
        return True

def run_prediction_loop(strategy, symbols):
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼")
    sys.stdout.flush()
    try:
        evaluate_predictions(get_price_now)
    except Exception as e:
        print(f"[ERROR] 평가 실패: {e}")
        sys.stdout.flush()

    all_results = []
    try:
        failure_map = load_failure_count()
    except Exception:
        failure_map = {}

    for symbol in symbols:
        try:
            if not should_predict(symbol, strategy):
                continue

            result = predict(symbol, strategy)
            print(f"[예측] {symbol}-{strategy} → {result}")
            sys.stdout.flush()

            if result.get("reason") in ["모델 없음", "데이터 부족", "feature 부족"]:
                print(f"[SKIP] {symbol}-{strategy} → 예측 불가 이유: {result['reason']}")
                log_prediction(
                    symbol=symbol,
                    strategy=strategy,
                    direction="N/A",
                    entry_price=0,
                    target_price=0,
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    confidence=0.0,
                    model="unknown",
                    success=False,
                    reason=result["reason"],
                )
                log_audit(symbol, strategy, result, result["reason"])
                continue

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
            except:
                pass

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

    for res in final:
        try:
            msg = format_message(res)
            send_message(msg)

            try:
                with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    if os.stat(MESSAGE_LOG).st_size == 0:
                        writer.writerow(["timestamp", "symbol", "strategy", "message"])
                    writer.writerow([datetime.datetime.utcnow().isoformat(), res["symbol"], res["strategy"], msg])
            except Exception as e:
                print(f"[ERROR] 메시지 로그 기록 실패: {e}")

            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%} | 성공률: {res['success_rate']:.2f}")
        except Exception as e:
            print(f"[ERROR] 메시지 전송 실패: {e}")

def main(strategy=None):
    print(">>> [main] recommend.py 실행")
    sys.stdout.flush()

    if strategy:
        symbols = get_symbols_by_volatility(strategy)
        run_prediction_loop(strategy, symbols)
    else:
        for strategy in ["단기", "중기", "장기"]:
            symbols = get_symbols_by_volatility(strategy)
            run_prediction_loop(strategy, symbols)

def start_regular_prediction_loop():
    def loop():
        while True:
            for strategy in ["단기", "중기", "장기"]:
                print(f"[정기예측] {strategy} 전체 예측 실행")
                run_prediction_loop(strategy, SYMBOLS)
            time.sleep(3600)
    threading.Thread(target=loop, daemon=True).start()
