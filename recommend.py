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
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼"); sys.stdout.flush()
    results, fmap = [], load_failure_count()
    triggered_trainings = set()
    class_distribution = {}

    for item in symbols:
        symbol = item["symbol"]
        vol = item.get("volatility", 0)

        if not allow_prediction:
            log_audit(symbol, strategy, "예측 생략", f"예측 차단됨 (source={source})")
            continue

        try:
            model_count = len([
                f for f in os.listdir("/persistent/models")
                if f.startswith(f"{symbol}_{strategy}_") and (f.endswith(".pt") or f.endswith(".meta.json"))
            ])
            if model_count == 0:
                log_audit(symbol, strategy, None, "모델 없음")
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
                    source=source,
                    predicted_class=-1,
                    label=-1
                )
                continue

            pred_results = predict(symbol, strategy, source=source)
            if not isinstance(pred_results, list):
                pred_results = [pred_results]

            for result in pred_results:
                if not isinstance(result, dict) or result.get("reason") in ["모델 없음", "데이터 부족", "feature 부족"]:
                    reason = result.get("reason", "예측 실패") if isinstance(result, dict) else "predict() 반환 오류"
                    pred_class_val = -1
                    log_prediction(
                        symbol=symbol,
                        strategy=strategy,
                        direction="예측실패",
                        entry_price=0,
                        target_price=0,
                        timestamp=now_kst().isoformat(),
                        model=result.get("model", "unknown") if isinstance(result, dict) else "unknown",
                        success=False,
                        reason=reason,
                        rate=0.0,
                        return_value=0.0,
                        volatility=False,
                        source=source,
                        predicted_class=pred_class_val,
                        label=pred_class_val
                    )
                    log_audit(symbol, strategy, result, reason)
                    continue

                result["volatility"] = vol
                result["return"] = result.get("expected_return", 0.0)
                result["source"] = result.get("source", source)
                result["predicted_class"] = result.get("class", -1)
                pred_class_val = result.get("class", -1)

                log_prediction(
                    symbol=result.get("symbol", symbol),
                    strategy=result.get("strategy", strategy),
                    direction=f"class-{pred_class_val}",
                    entry_price=result.get("price", 0),
                    target_price=result.get("price", 0) * (1 + result.get("expected_return", 0)),
                    timestamp=result.get("timestamp", now_kst().isoformat()),
                    model=result.get("model", "unknown"),
                    success=True,
                    reason=result.get("reason", "예측 성공"),
                    rate=result.get("expected_return", 0.0),
                    return_value=result.get("expected_return", 0.0),
                    volatility=vol > 0,
                    source=result.get("source", source),
                    predicted_class=pred_class_val,
                    label=pred_class_val
                )
                log_audit(symbol, strategy, result, "예측 기록 완료")

                if pred_class_val != -1:
                    class_distribution.setdefault(f"{symbol}-{strategy}", []).append(pred_class_val)

                fmap[f"{symbol}-{strategy}"] = 0
                results.append(result)

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            log_audit(symbol, strategy, None, f"예측 예외: {e}")

    save_failure_count(fmap)

    # ✅ 예측 후 즉시 평가 제거
    # 평가 실행은 별도 스케줄러(cron) 또는 루프에서 전략별 시간 후 자동 실행됨
    # try:
    #     print("[평가 실행] evaluate_predictions 호출")
    #     from predict import evaluate_predictions
    #     evaluate_predictions(get_kline_by_strategy)
    # except Exception as e:
    #     print(f"[ERROR] 평가 실패: {e}")

def run_prediction(symbol, strategy):
    print(f">>> [run_prediction] {symbol} - {strategy} 예측 시작")

    MODEL_DIR = "/persistent/models"
    for mt in ["transformer", "cnn_lstm", "lstm"]:
        pt_file = f"{symbol}_{strategy}_{mt}.pt"
        meta_file = f"{symbol}_{strategy}_{mt}.meta.json"
        if os.path.exists(os.path.join(MODEL_DIR, pt_file)) and os.path.exists(os.path.join(MODEL_DIR, meta_file)):
            run_prediction_loop(strategy, [{"symbol": symbol, "model_type": mt}], source="단일", allow_prediction=True)
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
        predicted_class=-1,
        label=-1
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

if __name__ == "__main__":
    main()
    from train import train_models
    train_models(["BTCUSDT", "ETHUSDT"])  # ✅ 학습할 심볼 리스트

    
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

