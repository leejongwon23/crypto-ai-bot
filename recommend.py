import os, csv, sys, time, threading, datetime, pytz
from telegram_bot import send_message
from predict import predict
from logger import log_prediction, get_model_success_rate, get_actual_success_rate, get_strategy_eval_count, get_min_gain
from data.utils import SYMBOLS, get_kline_by_strategy
from src.message_formatter import format_message
import train
from model_weight_loader import model_exists

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

def should_predict(symbol, strategy):
    try:
        return get_model_success_rate(symbol, strategy, "ensemble") < 0.85 or get_strategy_eval_count(strategy) < 10
    except: return True

def run_prediction_loop(strategy, symbols):
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼"); sys.stdout.flush()
    results, fmap = [], load_failure_count()

    for item in symbols:
        symbol = item["symbol"]
        vol = item.get("volatility", 0)
        try:
            model_count = len([f for f in os.listdir("/persistent/models") if f.startswith(f"{symbol}_{strategy}_") and f.endswith(".pt")])
            if model_count == 0:
                r = get_min_gain(symbol, strategy)
                log_prediction(symbol, strategy, "N/A", 0, 0, now_kst().isoformat(),
                               0.0, "ensemble", False, "모델 없음", r, return_value=r)
                log_audit(symbol, strategy, None, "모델 없음")
                continue
            if not should_predict(symbol, strategy): continue
            result = predict(symbol, strategy)
            print(f"[예측] {symbol}-{strategy} → {result}"); sys.stdout.flush()
            if not isinstance(result, dict) or result.get("reason") in ["모델 없음", "데이터 부족", "feature 부족"]:
                reason = result.get("reason", "예측 실패") if isinstance(result, dict) else "predict() 반환 오류"
                r = get_min_gain(symbol, strategy)
                log_prediction(symbol, strategy, "N/A", 0, 0, now_kst().isoformat(),
                               0.0, "ensemble", False, reason, r, return_value=r)
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
                timestamp=now_kst().isoformat(),
                confidence=result.get("confidence", 0.0),
                model=result.get("model", "ensemble"),
                success=True,
                reason=result.get("reason", "예측 성공"),
                rate=result.get("rate", 0.0),
                return_value=result.get("return", 0.0)
            )
            log_audit(symbol, strategy, result, "예측 성공")

            key = f"{symbol}-{strategy}"
            if not result.get("success", False):
                fmap[key] = fmap.get(key, 0) + 1
                if fmap[key] >= FAIL_LIMIT:
                    print(f"[학습 트리거] {symbol}-{strategy} 실패 {fmap[key]}회 → 학습")
                    threading.Thread(target=train.train_model, args=(symbol, strategy), daemon=True).start()
                    fmap[key] = 0
            else:
                fmap[key] = 0
            results.append(result)

        except Exception as e:
            r = get_min_gain(symbol, strategy)
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            log_prediction(symbol, strategy, "예외", 0, 0, now_kst().isoformat(), 0.0, "ensemble", False,
                           f"예측 예외: {e}", r, return_value=r)
            log_audit(symbol, strategy, None, f"예측 예외: {e}")

    save_failure_count(fmap)

    filtered = [r for r in results if r.get("rate", 0.0) >= 0.01]
    final = sorted(filtered, key=lambda x: -x["rate"])[:SEND_LIMIT]

    for res in final:
        try:
            msg = ("[반전 추천] " if res.get("reversed") else "") + format_message(res)
            send_message(msg)
            with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([now_kst().isoformat(), res["symbol"], res["strategy"], msg])
            print(f"✅ 메시지 전송: {res['symbol']}-{res['strategy']} → {res['direction']} | 수익률: {res['rate']:.2%}")
        except Exception as e:
            print(f"[ERROR] 메시지 전송 실패: {e}")
            with open(MESSAGE_LOG, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([now_kst().isoformat(), res["symbol"], res["strategy"], f"전송 실패: {e}"])

def run_prediction(symbol, strategy):
    print(f">>> [run_prediction] {symbol} - {strategy} 예측 시작")
    run_prediction_loop(strategy, [{"symbol": symbol}])

def main(strategy=None):
    print(">>> [main] recommend.py 실행")
    targets = [strategy] if strategy else ["단기", "중기", "장기"]
    for s in targets:
        run_prediction_loop(s, get_symbols_by_volatility(s))
