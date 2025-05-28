import os, csv, sys, time, threading, datetime, pytz
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, strategy_stats, get_strategy_eval_count
from data.utils import SYMBOLS, get_kline_by_strategy
from src.message_formatter import format_message
import train
from model_weight_loader import get_model_weight

FAIL_LIMIT, SEND_LIMIT = 3, 5
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
            if df is None or len(df) < 60: continue
            r_std = df["close"].pct_change().rolling(20).std().iloc[-1]
            b_std = df["close"].pct_change().rolling(60).std().iloc[-1]
            if r_std >= th and r_std / (b_std + 1e-8) >= 1.5:
                result.append({"symbol": symbol, "volatility": r_std})
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return sorted(result, key=lambda x: -x["volatility"])[:30]

def run_prediction_loop(strategy, symbols):
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼"); sys.stdout.flush()
    results, fmap = [], load_failure_count()

    for item in symbols:
        symbol = item["symbol"]
        vol = item.get("volatility", 0)
        try:
            model_count = len([f for f in os.listdir("/persistent/models") if f.startswith(f"{symbol}_{strategy}_") and f.endswith(".pt")])
            if model_count == 0:
                log_audit(symbol, strategy, None, "모델 없음")
                continue

            pred_results = predict(symbol, strategy)
            if not isinstance(pred_results, list):
                pred_results = [pred_results]

            for result in pred_results:
                if not isinstance(result, dict) or result.get("reason") in ["모델 없음", "데이터 부족", "feature 부족"]:
                    reason = result.get("reason", "예측 실패") if isinstance(result, dict) else "predict() 반환 오류"
                    log_prediction(symbol, strategy, "N/A", 0, 0, now_kst().isoformat(),
                                   model=result.get("model", "unknown"),
                                   success=False, reason=reason,
                                   rate=0.0, return_value=0.0, volatility=False)
                    log_audit(symbol, strategy, result, reason)
                    continue

                result["volatility"] = vol
                result["return"] = result.get("rate", 0.0)

                log_prediction(
                    symbol=result.get("symbol", symbol),
                    strategy=result.get("strategy", strategy),
                    direction=result.get("direction", "예측실패"),
                    entry_price=result.get("price", 0),
                    target_price=result.get("target", 0),
                    timestamp=result.get("timestamp", now_kst().isoformat()),
                    model=result.get("model", "unknown"),
                    success=result.get("success", True),
                    reason=result.get("reason", "예측 성공"),
                    rate=result.get("rate", 0.0),
                    return_value=result.get("return", 0.0),
                    volatility=vol > 0
                )
                log_audit(symbol, strategy, result, "예측 성공")

                key = f"{symbol}-{strategy}"
                if not result.get("success", False):
                    print(f"[오답학습 트리거] {symbol}-{strategy} → 예측 실패 감지 → 즉시 학습 실행")
                    threading.Thread(target=train.train_model, args=(symbol, strategy), daemon=True).start()
                fmap[key] = 0
                results.append(result)

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            log_audit(symbol, strategy, None, f"예측 예외: {e}")

    save_failure_count(fmap)

    # 필터 1단계: 수익률 기준 필터링
    strat_return = {}
    for r in results:
        strat_return.setdefault(r["strategy"], []).append(r)

    top_return = []
    for strat, items in strat_return.items():
        # 수익률 탑 5
        top5 = sorted(items, key=lambda x: -abs(x["rate"]))[:5]
        top_return.extend(top5)

    # 필터 2단계: 모델 가중치 최고
    weight_best = {}
    for r in top_return:
        strategy = r["strategy"]
        model = r["model"]
        symbol = r["symbol"]
        weight = get_model_weight(model, strategy, symbol)
        if strategy not in weight_best or weight > weight_best[strategy]["weight"]:
            weight_best[strategy] = {**r, "weight": weight}

    # 전송
    for strategy, result in weight_best.items():
        try:
            msg = format_message(result)
            send_message(msg)
            with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([now_kst().isoformat(), result["symbol"], result["strategy"], msg])
            print(f"✅ 메시지 전송: {result['symbol']} - {result['strategy']} - {result['direction']} | {result['rate']:.2%}")
        except Exception as e:
            print(f"[ERROR] 메시지 전송 실패: {e}")
            with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([now_kst().isoformat(), result["symbol"], result["strategy"], f"전송 실패: {e}"])

def run_prediction(symbol, strategy):
    print(f">>> [run_prediction] {symbol} - {strategy} 예측 시작")
    run_prediction_loop(strategy, [{"symbol": symbol}])

def main(strategy=None):
    print(">>> [main] recommend.py 실행")
    targets = [strategy] if strategy else ["단기", "중기", "장기"]
    for s in targets:
        run_prediction_loop(s, get_symbols_by_volatility(s))
