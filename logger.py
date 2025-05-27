import os, csv, datetime, pandas as pd, pytz
from data.utils import get_kline_by_strategy

DIR, LOG = "/persistent", "/persistent/logs"
PREDICTION_LOG, WRONG = f"{DIR}/prediction_log.csv", f"{DIR}/wrong_predictions.csv"
TRAIN_LOG, AUDIT_LOG = f"{LOG}/train_log.csv", f"{LOG}/evaluation_audit.csv"
STOP_LOSS = 0.02
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
model_success_tracker = {}
os.makedirs(LOG, exist_ok=True)

def get_min_gain(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < 20: return {"단기":0.01,"중기":0.03,"장기":0.05}.get(strategy,0.05)
    v = df["close"].pct_change().rolling(20).std().iloc[-1] if not df.empty else 0.01
    return max(round(v*1.2,4), {"단기":0.005,"중기":0.01,"장기":0.02}.get(strategy,0.03))

def update_model_success(s, t, m, success):
    k = (s, t or "알수없음", m)
    model_success_tracker.setdefault(k, {"success":0,"fail":0})
    model_success_tracker[k]["success" if success else "fail"] += 1

def get_model_success_rate(s, t, m, min_total=10):
    r = model_success_tracker.get((s, t or "알수없음", m), {"success":0,"fail":0})
    total = r["success"] + r["fail"]
    return 0.5 if total < min_total else r["success"] / total

def get_actual_success_rate(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        df = df[df["status"].isin(["success", "fail"])]
        return round(len(df[df["status"] == "success"]) / len(df), 4) if len(df) > 0 else 0.0
    except: return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return len(df[(df["strategy"] == strategy) & df["status"].isin(["success", "fail"])])
    except: return 0

def log_audit(s, t, status, reason):
    row = {
        "timestamp": now_kst().isoformat(),
        "symbol": str(s or "UNKNOWN"),
        "strategy": str(t or "알수없음"),
        "status": str(status),
        "reason": str(reason)
    }
    header = not os.path.exists(AUDIT_LOG) or os.stat(AUDIT_LOG).st_size == 0
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if header: w.writeheader()
            w.writerow(row)
    except: pass

def log_prediction(symbol, strategy, direction=None, entry_price=0, target_price=0,
                   timestamp=None, model=None, success=True, reason="", rate=0.0,
                   return_value=None, volatility=False):
    now = timestamp or now_kst().isoformat()
    mname = str(model or "unknown")
    row = {
        "timestamp": now,
        "symbol": str(symbol or "UNKNOWN"),
        "strategy": str(strategy or "알수없음"),
        "direction": direction or "N/A",
        "entry_price": float(entry_price),
        "target_price": float(target_price),
        "model": mname,
        "rate": float(rate),
        "status": "pending" if success else "failed",
        "reason": reason or "",
        "return": float(return_value if return_value is not None else rate),
        "volatility": bool(volatility)
    }
    log_audit(row["symbol"], row["strategy"], "예측성공" if success else "예측실패", row["reason"])
    try:
        with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not os.path.exists(PREDICTION_LOG) or os.stat(PREDICTION_LOG).st_size == 0:
                w.writeheader()
            w.writerow(row)
    except: pass

def log_training_result(symbol, strategy, model_name, acc, f1, loss):
    row = {
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "strategy": strategy,
        "model": model_name,
        "accuracy": acc,
        "f1_score": f1,
        "loss": loss
    }
    try:
        pd.DataFrame([row]).to_csv(TRAIN_LOG, mode="a", index=False,
                                   header=not os.path.exists(TRAIN_LOG),
                                   encoding="utf-8-sig")
    except: pass

def get_dynamic_eval_wait(s):
    return {"단기":4, "중기":24, "장기":168}.get(s, 6)

def evaluate_predictions(get_price_fn):
    if not os.path.exists(PREDICTION_LOG): return
    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
    except: return
    now, updated = now_kst(), []
    for r in rows:
        try:
            if r.get("status") not in ["pending", "failed"]:
                updated.append(r); continue
            s, strat = r["symbol"], r["strategy"]
            d, m = r.get("direction", "롱"), r.get("model", "unknown")
            entry, rate = float(r.get("entry_price", 0)), float(r.get("rate", 0))
            pred_time = datetime.datetime.fromisoformat(r["timestamp"]).astimezone(pytz.timezone("Asia/Seoul"))
            hours = (now - pred_time).total_seconds() / 3600
            vol = str(r.get("volatility", "False")).lower() in ["1", "true", "yes"]
            df = get_kline_by_strategy(s, strat)
            if hours < get_dynamic_eval_wait(strat):
                r.update({"reason": f"{hours:.2f}h < {get_dynamic_eval_wait(strat)}h", "return": 0.0})
            elif entry == 0 or m == "unknown" or any(k in r.get("reason", "") for k in ["모델 없음", "기준 미달"]):
                r.update({"status": "v_invalid_model" if vol else "invalid_model", "reason": "모델 없음 또는 entry=0", "return": 0.0})
            elif df is None or df.empty or df[df["timestamp"] >= pred_time].empty:
                r.update({"status": "skip_eval", "reason": "평가 데이터 부족", "return": 0.0})
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
                eval_df = df[df["timestamp"] >= pred_time]
                price = eval_df["high"].max() if d == "롱" else eval_df["low"].min()
                gain = (price - entry) / entry if d == "롱" else (entry - price) / entry
                success = gain >= rate
                r.update({
                    "status": "v_success" if vol and success else "v_fail" if vol else "success" if success else "fail",
                    "reason": f"도달: {gain:.4f} ≥ {rate:.4f}" if success else f"미달: {gain:.4f} < {rate:.4f}",
                    "return": round(gain, 4)
                })
                update_model_success(s, strat, m, success)
                audit_row = {
                    "timestamp": now.isoformat(), "symbol": s, "strategy": strat,
                    "model": m, "status": r.get("status", ""), "reason": r.get("reason", ""),
                    "predicted_return": rate, "actual_return": round(gain, 4),
                    "accuracy_before": "", "accuracy_after": "",
                    "predicted_volatility": float(rate) if vol else "",
                    "actual_volatility": eval_df["close"].pct_change().rolling(5).std().iloc[-1] if not eval_df.empty else ""
                }
                write_header = not os.path.exists(AUDIT_LOG) or os.stat(AUDIT_LOG).st_size == 0
                with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as af:
                    writer = csv.DictWriter(af, fieldnames=audit_row.keys())
                    if write_header: writer.writeheader()
                    writer.writerow(audit_row)
                if not success:
                    with open(WRONG, "a", newline="", encoding="utf-8-sig") as wf:
                        csv.writer(wf).writerow([r["timestamp"], s, strat, d, entry, r.get("target_price", 0), gain])
        except Exception as e:
            r.update({"status": "skip_eval", "reason": f"예외 발생: {e}", "return": 0.0})
        updated.append(r)
    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=updated[0].keys())
        w.writeheader(); w.writerows(updated)

strategy_stats = {}
