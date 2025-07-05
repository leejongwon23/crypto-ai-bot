import os, csv, sys, time, threading, datetime, pytz
import json
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, strategy_stats, get_strategy_eval_count
from data.utils import SYMBOLS, get_kline_by_strategy
from src.message_formatter import format_message
import train

STRATEGY_VOL = {"단기": 0.003, "중기": 0.005, "장기": 0.008}
AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
MESSAGE_LOG = "/persistent/logs/message_log.csv"
os.makedirs("/persistent/logs", exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def log_audit(symbol, strategy, result, status):
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "symbol", "strategy", "result", "status"])
            if f.tell() == 0: w.writeheader()
            w.writerow({
                "timestamp": now_kst().isoformat(),
                "symbol": symbol or "UNKNOWN",
                "strategy": strategy or "알수없음",
                "result": str(result),
                "status": status
            })
    except Exception as e:
        print(f"[log_audit 오류] {e}")

def load_failure_count():
    if not os.path.exists(FAILURE_LOG): return {}
    with open(FAILURE_LOG, "r", encoding="utf-8-sig") as f:
        return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}

def save_failure_count(fmap):
    with open(FAILURE_LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "strategy", "failures"])
        w.writeheader()
        for k, v in fmap.items():
            s, strat = k.split("-")
            w.writerow({"symbol": s, "strategy": strat, "failures": v})

def get_symbols_by_volatility(strategy):
    th = STRATEGY_VOL.get(strategy, 0.003)
    result = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 60:
                continue
            r_std = df["close"].pct_change().rolling(20).std().iloc[-1]
            b_std = df["close"].pct_change().rolling(60).std().iloc[-1]
            is_volatile = r_std >= th
            is_rising = (r_std / (b_std + 1e-8)) >= 1.2
            if is_volatile and is_rising:
                result.append({"symbol": symbol, "volatility": r_std})
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return sorted(result, key=lambda x: -x["volatility"])

def run_prediction_loop(strategy, symbols, source="일반", allow_prediction=True):
    import sys
    from predict import predict
    from logger import log_prediction
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼"); sys.stdout.flush()
    results = []

    for item in symbols:
        symbol = item["symbol"]

        if not allow_prediction:
            continue

        try:
            pred_results = predict(symbol, strategy, source=source)
            if not isinstance(pred_results, list):
                pred_results = [pred_results]

            for result in pred_results:
                pred_class_val = result.get("class", -1) if isinstance(result, dict) else -1

                log_prediction(
                    symbol=result.get("symbol", symbol) if isinstance(result, dict) else symbol,
                    strategy=strategy,
                    direction=f"class-{pred_class_val}",
                    entry_price=result.get("price", 0) if isinstance(result, dict) else 0,
                    target_price=result.get("price", 0) * (1 + result.get("expected_return", 0)) if isinstance(result, dict) else 0,
                    timestamp=result.get("timestamp") if isinstance(result, dict) else None,
                    model=result.get("model", "unknown") if isinstance(result, dict) else "unknown",
                    success=True,
                    reason=result.get("reason", "예측 성공") if isinstance(result, dict) else "예측 실패",
                    rate=result.get("expected_return", 0.0) if isinstance(result, dict) else 0.0,
                    return_value=result.get("expected_return", 0.0) if isinstance(result, dict) else 0.0,
                    volatility=False,
                    source=source,
                    predicted_class=pred_class_val,
                    label=pred_class_val
                )
                results.append(result)

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

    # ✅ 평가 호출 부분 완전 삭제 완료

    return results

def run_prediction(symbol, strategy):
    print(f">>> [run_prediction] {symbol} - {strategy} 예측 시작")

    MODEL_DIR = "/persistent/models"
    for mt in ["transformer", "cnn_lstm", "lstm"]:
        pt_file = f"{symbol}_{strategy}_{mt}.pt"
        meta_file = f"{symbol}_{strategy}_{mt}.meta.json"
        if os.path.exists(os.path.join(MODEL_DIR, pt_file)) and os.path.exists(os.path.join(MODEL_DIR, meta_file)):
            # 🔧 [Diversity Regularization 추가]
            # run_prediction_loop 호출 전 diversity_penalty 파라미터 전달 (호출 구조 수정 필요)
            run_prediction_loop(strategy, [{"symbol": symbol, "model_type": mt}], source="단일", allow_prediction=True, diversity_penalty=True)
            return

    print(f"[run_prediction 오류] {symbol}-{strategy} 가능한 모델 없음")
    log_prediction(
        symbol=symbol,
        strategy=strategy,
        direction="예측실패",
        entry_price=0,
        target_price=0,
        timestamp=now_kst().isoformat(),
        model="unknown",
        success=False,
        reason="모델 없음",
        rate=0.0,
        return_value=0.0,
        volatility=False,
        source="단일",
        predicted_class=-1
    )

def main(strategy=None, symbol=None, force=False, allow_prediction=True):
    print(">>> [main] recommend.py 실행")
    check_disk_usage(threshold_percent=90)
    targets = [strategy] if strategy else ["단기", "중기", "장기"]
    from data.utils import SYMBOLS

    for s in targets:
        symbols_list = []
        if symbol:
            symbols_list.append({"symbol": symbol, "volatility": 0.0})
        else:
            for sym in SYMBOLS:
                symbols_list.append({"symbol": sym, "volatility": 0.0})
        run_prediction_loop(s, symbols_list, source="일반", allow_prediction=allow_prediction)

import shutil
def check_disk_usage(threshold_percent=90):
    try:
        total, used, free = shutil.disk_usage("/persistent")
        used_percent = (used / total) * 100
        if used_percent >= threshold_percent:
            print(f"🚨 경고: 디스크 사용량 {used_percent:.2f}% (한도 {threshold_percent}%) 초과")
        else:
            print(f"✅ 디스크 사용량 정상: {used_percent:.2f}%")
    except Exception as e:
        print(f"[디스크 사용량 확인 실패] {e}")

