import datetime
import os
import csv
import threading
import sys
import time
import pytz

from telegram_bot import send_message
from predict import predict
from logger import (
    log_prediction, get_model_success_rate,
    get_actual_success_rate, get_strategy_eval_count,
    get_min_gain
)
from data.utils import SYMBOLS, get_kline_by_strategy
from src.message_formatter import format_message
import train
from model_weight_loader import model_exists

MIN_CONFIDENCE = 0.70
REVERSAL_CONF_THRESHOLD = 0.45
REVERSAL_SUCCESS_THRESHOLD = 0.6
SUCCESS_RATE_THRESHOLD = 0.70
FAILURE_TRIGGER_LIMIT = 3
MIN_SCORE_THRESHOLD = 0.005
FINAL_SEND_LIMIT = 5

STRATEGY_VOLATILITY = {"단기": 0.003, "중기": 0.005, "장기": 0.008}

AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
MESSAGE_LOG = "/persistent/logs/message_log.csv"
os.makedirs("/persistent/logs", exist_ok=True)

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def load_failure_count():
    if not os.path.exists(FAILURE_LOG): return {}
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
    now = now_kst().isoformat()
    row = {
        "timestamp": now,
        "symbol": symbol,
        "strategy": strategy,
        "result": str(result),
        "status": status,
    }
    write_header = not os.path.exists(AUDIT_LOG)
    with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: writer.writeheader()
        writer.writerow(row)

def get_symbols_by_volatility(strategy):
    threshold = STRATEGY_VOLATILITY.get(strategy, 0.003) * 1.2
    selected = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 20: continue
            vol = df["close"].pct_change().rolling(window=20).std().iloc[-1]
            if vol and vol >= threshold:
                selected.append({"symbol": symbol, "volatility": vol})
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return selected

def should_predict(symbol, strategy):
    try:
        rate = get_model_success_rate(symbol, strategy, "ensemble")
        eval_count = get_strategy_eval_count(strategy)
        return rate < 0.85 or eval_count < 10
    except:
        return True

def run_prediction_loop(strategy, symbol_data_list):
    print(f"[예측 시작 - {strategy}] {len(symbol_data_list)}개 심볼")
    sys.stdout.flush()

    all_results = []
    failure_map = load_failure_count()

    for item in symbol_data_list:
        symbol = item["symbol"]
        volatility = item.get("volatility", 0)
        try:
            if not model_exists(symbol, strategy):
                min_gain = get_min_gain(symbol, strategy)
                log_prediction(symbol, strategy, "N/A", 0, 0, now_kst().isoformat(), 0.0, "ensemble", False, "모델 없음", min_gain)
                log_audit(symbol, strategy, None, "모델 없음")
                continue

            if not should_predict(symbol, strategy):
                continue

            result = predict(symbol, strategy)
            print(f"[예측] {symbol}-{strategy} → {result}")
            sys.stdout.flush()

            if not isinstance(result, dict) or result.get("reason") in ["모델 없음", "데이터 부족", "feature 부족"]:
                reason = result.get("reason", "예측 실패") if isinstance(result, dict) else "predict() 반환 오류"
                min_gain = get_min_gain(symbol, strategy)
                log_prediction(symbol, strategy, "N/A", 0, 0, now_kst().isoformat(), 0.0, "ensemble", False, reason, min_gain)
                log_audit(symbol, strategy, result, reason)
                continue

            result["volatility"] = volatility
            log_prediction(
                symbol=result.get("symbol", symbol),
                strategy=result.get("strategy", strategy),
                direction=result.get("direction", "예측실패"),
                entry_price=result.get("price", 0),
                target_price=result.get("target", 0),
                timestamp=now_kst().isoformat(),
                confidence=result.get("confidence", 0.0),
                model=result.get("model", "ensemble"),
                success=True,
                reason=result.get("reason", "예측 성공"),
                rate=result.get("rate", get_min_gain(symbol, strategy))
            )
            log_audit(symbol, strategy, result, "예측 성공")

            key = f"{symbol}-{strategy}"
            if not result.get("success", False):
                failure_map[key] = failure_map.get(key, 0) + 1
                if failure_map[key] >= FAILURE_TRIGGER_LIMIT:
                    print(f"[학습 트리거] {symbol}-{strategy} 실패 {failure_map[key]}회 → 학습")
                    threading.Thread(target=train.train_model, args=(symbol, strategy), daemon=True).start()
                    failure_map[key] = 0
            else:
                failure_map[key] = 0

            all_results.append(result)

            # ✅ 반전 전략 조건 판단
            conf = result.get("confidence", 0)
            rate = result.get("rate", 0)
            success_rate = get_model_success_rate(symbol, strategy, result.get("model", "ensemble"))
            if conf < REVERSAL_CONF_THRESHOLD and rate < get_min_gain(symbol, strategy) and success_rate < REVERSAL_SUCCESS_THRESHOLD:
                reversed_result = dict(result)
                reversed_result["direction"] = "숏" if result["direction"] == "롱" else "롱"
                reversed_result["confidence"] = 1 - conf
                reversed_result["rate"] = get_min_gain(symbol, strategy) * 1.1
                reversed_result["target"] = reversed_result["price"] * (1 + reversed_result["rate"]) if reversed_result["direction"] == "롱" else reversed_result["price"] * (1 - reversed_result["rate"])
                reversed_result["stop"] = reversed_result["price"] * (1 - 0.02) if reversed_result["direction"] == "롱" else reversed_result["price"] * (1 + 0.02)
                reversed_result["reason"] = "🔁 반전 전략: 낮은 신뢰도·낮은 수익률·낮은 성공률"
                reversed_result["reversed"] = True
                reversed_result["success_rate"] = success_rate
                all_results.append(reversed_result)

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            min_gain = get_min_gain(symbol, strategy)
            log_prediction(symbol, strategy, "예외", 0, 0, now_kst().isoformat(), 0.0, "ensemble", False, f"예외 발생: {e}", min_gain)
            log_audit(symbol, strategy, None, f"예외 발생: {e}")

    save_failure_count(failure_map)

    filtered = []
    for r in all_results:
        conf = r.get("confidence", 0)
        model = r.get("model", "")
        rate = r.get("rate", 0)
        vol = r.get("volatility", 0)
        symbol = r.get("symbol")
        strategy = r.get("strategy")
        success_rate = r.get("success_rate", get_model_success_rate(symbol, strategy, model))
        if conf < MIN_CONFIDENCE and not r.get("reversed"): continue
        if rate < get_min_gain(symbol, strategy): continue
        if success_rate < SUCCESS_RATE_THRESHOLD: continue

        score = (conf ** 1.5) * (rate ** 1.2) * (success_rate ** 1.2) * (1 + vol)
        if score < MIN_SCORE_THRESHOLD: continue

        r["success_rate"] = success_rate
        r["score"] = score
        filtered.append(r)

    final = sorted(filtered, key=lambda x: -x["score"])[:FINAL_SEND_LIMIT]

    for res in final:
        try:
            msg = ("[반전 추천] " if res.get("reversed") else "") + format_message(res)
            send_message(msg)
            with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([now_kst().isoformat(), res["symbol"], res["strategy"], msg])
            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%} | 성공률: {res['success_rate']:.2f}")
        except Exception as e:
            print(f"[ERROR] 메시지 전송 실패: {e}")
            with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([now_kst().isoformat(), res["symbol"], res["strategy"], f"전송 실패: {e}"])

def run_prediction(symbol, strategy):
    print(f">>> [run_prediction] {symbol} - {strategy} 예측 시작")
    run_prediction_loop(strategy, [{"symbol": symbol}])

def main(strategy=None):
    print(">>> [main] recommend.py 실행")
    if strategy:
        run_prediction_loop(strategy, get_symbols_by_volatility(strategy))
    else:
        for strat in ["단기", "중기", "장기"]:
            run_prediction_loop(strat, get_symbols_by_volatility(strat))
