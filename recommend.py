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
from data.utils import SYMBOLS, get_realtime_prices, get_kline_by_strategy
from src.message_formatter import format_message
import train
from model_weight_loader import model_exists

# --- 설정 상수 ---
MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_OVERRIDE = 0.85
SUCCESS_RATE_THRESHOLD = 0.70
FAILURE_TRIGGER_LIMIT = 3
MIN_SCORE_THRESHOLD = 0.005
FINAL_SEND_LIMIT = 5
VOLATILITY_THRESHOLD = 0.003

# --- 경로 설정 ---
AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
MESSAGE_LOG = "/persistent/logs/message_log.csv"
os.makedirs("/persistent/logs", exist_ok=True)

# --- 시간 함수 (서울 기준) ---
def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# --- 실패 횟수 로딩 및 저장 ---
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

# --- 감사 로그 기록 ---
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

# --- 실시간 가격 조회 ---
def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

# --- 변동성 기준 필터링 ---
def get_symbols_by_volatility(strategy, threshold=VOLATILITY_THRESHOLD):
    threshold *= 1.2
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

# --- 예측 실행 여부 판단 ---
def should_predict(symbol, strategy):
    try:
        rate = get_model_success_rate(symbol, strategy, "ensemble")
        eval_count = get_strategy_eval_count(strategy)
        return rate < 0.85 or eval_count < 10
    except:
        return True

# --- 예측 루프 실행 ---
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
                log_prediction(symbol, strategy, "N/A", 0, 0, now_kst().isoformat(), 0.0, "unknown", False, "모델 없음", 0)
                log_audit(symbol, strategy, None, "모델 없음")
                continue

            if not should_predict(symbol, strategy):
                continue

            result = predict(symbol, strategy)
            print(f"[예측] {symbol}-{strategy} → {result}")
            sys.stdout.flush()

            if not isinstance(result, dict) or result.get("reason") in ["모델 없음", "데이터 부족", "feature 부족"]:
                reason = result.get("reason", "예측 실패") if isinstance(result, dict) else "predict() 반환 오류"
                log_prediction(symbol, strategy, "N/A", 0, 0, now_kst().isoformat(), 0.0, "unknown", False, reason, 0)
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
                model=result.get("model", "unknown"),
                success=True,
                reason=result.get("reason", "예측 성공"),
                rate=result.get("rate", 0.0)
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

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            log_prediction(symbol, strategy, "예외", 0, 0, now_kst().isoformat(), 0.0, "unknown", False, f"예외 발생: {e}", 0)
            log_audit(symbol, strategy, None, f"예외 발생: {e}")

    save_failure_count(failure_map)

    # --- 결과 필터링 ---
    filtered = []
    for r in all_results:
        conf = r.get("confidence", 0)
        model = r.get("model", "")
        rate = r.get("rate", 0)
        vol = r.get("volatility", 0)
        symbol = r.get("symbol")
        strategy = r.get("strategy")

        success_rate = get_model_success_rate(symbol, strategy, model)
        if conf < MIN_CONFIDENCE: continue
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
            msg = format_message(res)
            send_message(msg)
            with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([now_kst().isoformat(), res["symbol"], res["strategy"], msg])
            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%} | 성공률: {res['success_rate']:.2f}")
        except Exception as e:
            print(f"[ERROR] 메시지 전송 실패: {e}")

# --- 단일 예측 실행 ---
def run_prediction(symbol, strategy):
    print(f">>> [run_prediction] {symbol} - {strategy} 예측 시작")
    run_prediction_loop(strategy, [{"symbol": symbol}])

# --- 전략별 전체 실행 ---
def main(strategy=None):
    print(">>> [main] recommend.py 실행")
    if strategy:
        run_prediction_loop(strategy, get_symbols_by_volatility(strategy))
