import os, csv, datetime, pandas as pd, pytz
from data.utils import get_kline_by_strategy

DIR, LOG = "/persistent", "/persistent/logs"
PREDICTION_LOG, WRONG = f"{DIR}/prediction_log.csv", f"{DIR}/wrong_predictions.csv"
CORRECT = f"{DIR}/correct_predictions.csv"
EVAL_RESULT = f"{DIR}/evaluation_result.csv"  # ✅ 추가
TRAIN_LOG, AUDIT_LOG = f"{LOG}/train_log.csv", f"{LOG}/evaluation_audit.csv"
STOP_LOSS = 0.02
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
model_success_tracker = {}
os.makedirs(LOG, exist_ok=True)

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
                   return_value=None, volatility=False, source="일반"):
    now = timestamp or now_kst().isoformat()
    mname = str(model or "unknown")
    status = "v_pending" if volatility and success else "v_failed" if volatility and not success else "pending" if success else "failed"
    row = {
        "timestamp": now,
        "symbol": str(symbol or "UNKNOWN"),
        "strategy": str(strategy or "알수없음"),
        "direction": direction or "N/A",
        "entry_price": float(entry_price),
        "target_price": float(target_price),
        "model": mname,
        "rate": float(rate),
        "status": status,
        "reason": reason or "",
        "return": float(return_value if return_value is not None else rate),
        "volatility": bool(volatility),
        "source": source
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
        "accuracy": float(acc),
        "f1_score": float(f1),
        "loss": float(loss)
    }
    try:
        pd.DataFrame([row]).to_csv(TRAIN_LOG, mode="a", index=False,
                                   header=not os.path.exists(TRAIN_LOG),
                                   encoding="utf-8-sig")
    except: pass

def get_dynamic_eval_wait(s):
    return {"단기":4, "중기":24, "장기":168}.get(s, 6)

import hashlib

def get_feature_hash(feature_row):
    rounded = [round(float(x), 4) for x in feature_row]
    joined = ",".join(map(str, rounded))
    return hashlib.sha1(joined.encode()).hexdigest()


            
def evaluate_predictions(get_price_fn):
    from failure_db import ensure_failure_db, insert_failure_record, load_existing_failure_hashes
    ensure_failure_db()

    if not os.path.exists(PREDICTION_LOG):
        return
    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))

# ✅ 예측 로그가 비어 있는 경우 평가 건너뛰기
if not rows:
    print("[스킵] 예측 결과가 하나도 없어서 평가를 건너뜁니다.")
    return
    except:
        return

    now = now_kst()
    updated, evaluated = [], []
    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}  # 시간 단위

    for r in rows:
        try:
            if r.get("status") not in ["pending", "failed", "v_pending", "v_failed"]:
                updated.append(r)
                continue

            s, strat = r["symbol"], r["strategy"]
            d = r.get("direction", "롱")
            m = r.get("model", "unknown")
            entry = float(r.get("entry_price", 0))
            rate = float(r.get("rate", 0))
            pred_time = datetime.datetime.fromisoformat(r["timestamp"]).astimezone(pytz.timezone("Asia/Seoul"))
            eval_deadline = pred_time + datetime.timedelta(hours=eval_horizon_map.get(strat, 6))
            vol = str(r.get("volatility", "False")).lower() in ["1", "true", "yes"]

            df = get_price_fn(s, strat)

            if now < eval_deadline:
                r.update({
                    "reason": f"⏳ 평가 대기 중 ({now.strftime('%H:%M')} < {eval_deadline.strftime('%H:%M')})",
                    "return": 0.0
                })
            elif entry == 0 or m == "unknown" or any(k in r.get("reason", "") for k in ["모델 없음", "기준 미달"]):
                r.update({
                    "status": "v_invalid_model" if vol else "invalid_model",
                    "reason": "모델 없음 또는 entry=0",
                    "return": 0.0
                })
            elif df is None or df.empty or df[df["timestamp"] >= pred_time].empty:
                r.update({
                    "status": "skip_eval",
                    "reason": "평가 데이터 부족",
                    "return": 0.0
                })
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
                eval_df = df[(df["timestamp"] >= pred_time) & (df["timestamp"] <= eval_deadline)]
                price = eval_df["high"].max() if d == "롱" else eval_df["low"].min()
                gain = (price - entry) / entry if d == "롱" else (entry - price) / entry
                success = gain >= rate

                r.update({
                    "status": "v_success" if vol and success else "v_fail" if vol else "success" if success else "fail",
                    "reason": f"도달: {gain:.4f} ≥ {rate:.4f}" if success else f"미달: {gain:.4f} < {rate:.4f}",
                    "return": float(round(gain, 4))
                })

                update_model_success(s, strat, m, success)
                evaluated.append(dict(r))

        except Exception as e:
            r.update({
                "status": "skip_eval",
                "reason": f"예외 발생: {e}",
                "return": 0.0
            })
        updated.append(r)

    # ✅ 안전하게 평가결과 기록
    if evaluated and len(evaluated) > 0:
        with open(EVAL_RESULT, "a", newline="", encoding="utf-8-sig") as ef:
            w = csv.DictWriter(ef, fieldnames=evaluated[0].keys())
            if not os.path.exists(EVAL_RESULT) or os.stat(EVAL_RESULT).st_size == 0:
                w.writeheader()
            w.writerows(evaluated)

        failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
        if failed:
            with open(WRONG, "a", newline="", encoding="utf-8-sig") as wf:
                w = csv.DictWriter(wf, fieldnames=failed[0].keys())
                if not os.path.exists(WRONG) or os.stat(WRONG).st_size == 0:
                    w.writeheader()
                w.writerows(failed)

            try:
                existing_hashes = load_existing_failure_hashes()
                for r in failed:
                    symbol, strategy = r["symbol"], r["strategy"]
                    df = get_price_fn(symbol, strategy)
                    from data.utils import compute_features
                    df_feat = compute_features(symbol, df, strategy)
                    if df_feat is None or df_feat.empty:
                        continue
                    feature_row = df_feat.dropna().iloc[-1].values
                    from logger import get_feature_hash
                    hash_value = get_feature_hash(feature_row)
                    key = (symbol, strategy, r.get("direction", "예측실패"), hash_value)
                    if key in existing_hashes:
                        continue
                    insert_failure_record(r, hash_value)
                    existing_hashes.add(key)
            except Exception as e:
                print(f"[실패패턴 기록 오류] {e}")
                sys.stdout.flush()

    # ✅ 모든 항목 업데이트
    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=updated[0].keys())
        w.writeheader()
        w.writerows(updated)
                
strategy_stats = {}

