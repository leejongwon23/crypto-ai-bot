import os, csv, datetime, pandas as pd, pytz
from data.utils import get_kline_by_strategy

PERSIST_DIR = "/persistent"
LOG_DIR, MODEL_DIR = os.path.join(PERSIST_DIR, "logs"), os.path.join(PERSIST_DIR, "models")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
LOG_FILE, AUDIT_LOG = os.path.join(LOG_DIR, "train_log.csv"), os.path.join(LOG_DIR, "evaluation_audit.csv")
EVAL_EXPIRY_BUFFER, STOP_LOSS_PCT, model_success_tracker = 12, 0.02, {}
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
os.makedirs(LOG_DIR, exist_ok=True)

def get_min_gain(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < 20: return {"단기": 0.01, "중기": 0.03, "장기": 0.05}.get(strategy, 0.05)
    v = df["close"].pct_change().rolling(20).std().iloc[-1] if not df.empty else 0.01
    return max(round(v * 1.2, 4), {"단기": 0.005, "중기": 0.01, "장기": 0.02}.get(strategy, 0.03))

def update_model_success(symbol, strategy, model, success):
    key = (symbol, strategy or "알수없음", model)
    model_success_tracker.setdefault(key, {"success": 0, "fail": 0})
    model_success_tracker[key]["success" if success else "fail"] += 1

def get_model_success_rate(symbol, strategy, model, min_total=10):
    key = (symbol, strategy or "알수없음", model)
    r = model_success_tracker.get(key, {"success": 0, "fail": 0})
    total = r["success"] + r["fail"]
    return 0.5 if total < min_total else r["success"] / total

def get_actual_success_rate(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        df = df[df["status"].isin(["success", "fail"])]
        return round(len(df[df["status"] == "success"]) / len(df), 4) if len(df) > 0 else 0.0
    except Exception as e:
        print(f"[오류] get_actual_success_rate 실패: {e}"); return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return len(df[(df["strategy"] == strategy) & df["status"].isin(["success", "fail"])])
    except Exception as e:
        print(f"[오류] get_strategy_eval_count 실패: {e}"); return 0

def log_audit(symbol, strategy, status, reason):
    row = {"timestamp": now_kst().isoformat(), "symbol": str(symbol or "UNKNOWN"),
           "strategy": str(strategy or "알수없음"), "status": str(status), "reason": str(reason)}
    write_header = not os.path.exists(AUDIT_LOG) or os.stat(AUDIT_LOG).st_size == 0
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "symbol", "strategy", "status", "reason"])
            if write_header: w.writeheader(); w.writerow(row)
    except Exception as e:
        print(f"[오류] log_audit 실패: {e}")

def log_prediction(symbol, strategy, direction=None, entry_price=0, target_price=0,
                   timestamp=None, confidence=0, model="unknown", success=True,
                   reason="", rate=0.0, return_value=None, volatility=False):
    now = timestamp or now_kst().isoformat()
    row = {
        "timestamp": now, "symbol": str(symbol or "UNKNOWN"), "strategy": str(strategy or "알수없음"),
        "direction": direction or "N/A", "entry_price": float(entry_price), "target_price": float(target_price),
        "confidence": float(confidence), "model": model or "unknown", "rate": float(rate),
        "status": "pending" if success else "failed", "reason": reason or "",
        "return": float(return_value if return_value is not None else rate), "volatility": bool(volatility)
    }
    headers = ["timestamp","symbol","strategy","direction","entry_price","target_price","confidence",
               "model","rate","status","reason","return","volatility"]
    write_header = not os.path.exists(PREDICTION_LOG) or os.stat(PREDICTION_LOG).st_size == 0
    log_audit(row["symbol"], row["strategy"], "예측성공" if success else "예측실패", row["reason"])
    try:
        with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header: w.writeheader(); w.writerow(row)
    except Exception as e:
        print(f"[오류] log_prediction 실패: {e}")

def log_training_result(symbol, strategy, model_name, acc, f1, loss):
    row = {"timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
           "symbol": symbol, "strategy": strategy, "model": model_name,
           "accuracy": float(acc), "f1_score": float(f1), "loss": float(loss)}
    try:
        pd.DataFrame([row]).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE),
                                   index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[오류] 학습 로그 저장 실패: {e}")

def get_dynamic_eval_wait(strategy):
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 6)

def evaluate_predictions(get_price_fn):
    if not os.path.exists(PREDICTION_LOG): return
    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
    except Exception as e:
        print(f"[경고] 평가 로그 읽기 실패: {e}"); return
    now = now_kst(); updated = []
    headers = ["timestamp","symbol","strategy","direction","entry_price","target_price",
               "confidence","model","rate","status","reason","return","volatility"]
    for row in rows:
        try:
            if row.get("status") not in ["pending", "failed"]: updated.append(row); continue
            symbol, strategy = row.get("symbol", "UNKNOWN"), row.get("strategy", "알수없음")
            direction, model = row.get("direction", "롱"), row.get("model", "unknown")
            entry, rate = float(row.get("entry_price", 0)), float(row.get("rate", 0))
            pred_time = datetime.datetime.fromisoformat(row.get("timestamp")).astimezone(pytz.timezone("Asia/Seoul"))
            hours = (now - pred_time).total_seconds() / 3600
            volatility = str(row.get("volatility", "False")).lower() in ["1", "true", "yes"]
            if hours > get_dynamic_eval_wait(strategy) + EVAL_EXPIRY_BUFFER:
                row.update({"status": "v_expired" if volatility else "expired", "reason": f"평가 유효시간 초과: {hours:.2f}h", "return": 0.0})
            elif hours < get_dynamic_eval_wait(strategy):
                row.update({"reason": f"{hours:.2f}h < {get_dynamic_eval_wait(strategy)}h", "return": 0.0})
            elif entry == 0 or model == "unknown" or any(k in row.get("reason", "") for k in ["모델 없음", "기준 미달"]):
                row.update({"status": "v_invalid_model" if volatility else "invalid_model", "reason": "모델 없음 또는 entry_price=0 또는 기준 미달", "return": 0.0})
            else:
                df = get_kline_by_strategy(symbol, strategy)
                if df is None or df.empty or df[df["timestamp"] >= pred_time].empty:
                    row.update({"status": "skip_eval", "reason": "평가용 데이터 부족", "return": 0.0})
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
                    eval_df = df[df["timestamp"] >= pred_time]
                    price = eval_df["high"].max() if direction == "롱" else eval_df["low"].min()
                    gain = (price - entry) / entry if direction == "롱" else (entry - price) / entry
                    success = gain >= rate
                    row.update({
                        "status": "v_success" if volatility and success else "v_fail" if volatility and not success else "success" if success else "fail",
                        "reason": f"수익률 도달: {gain:.4f} ≥ 예측 {rate:.4f}" if success else f"미달: {gain:.4f} < 예측 {rate:.4f}",
                        "return": round(gain, 4)
                    })
                    update_model_success(symbol, strategy, model, success)
                    if not success:
                        if not os.path.exists(WRONG_PREDICTIONS):
                            with open(WRONG_PREDICTIONS, "w", newline="", encoding="utf-8-sig") as wf:
                                csv.writer(wf).writerow(["timestamp", "symbol", "strategy", "direction", "entry_price", "target_price", "gain"])
                        with open(WRONG_PREDICTIONS, "a", newline="", encoding="utf-8-sig") as wf:
                            csv.writer(wf).writerow([row["timestamp"], symbol, strategy, direction, entry, row.get("target_price", 0), gain])
            log_audit(symbol, strategy, row.get("status", "unknown"), row.get("reason", ""))
        except Exception as e:
            row.update({"status": "skip_eval", "reason": f"예외 발생: {e}", "return": 0.0})
            try: log_audit(row.get("symbol", "UNKNOWN"), row.get("strategy", "알수없음"), "스킵", row["reason"])
            except: pass
        updated.append(row)
    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader(); writer.writerows(updated)

strategy_stats = {}
