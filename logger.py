import os, csv, datetime, pandas as pd, pytz, hashlib
from data.utils import get_kline_by_strategy

DIR, LOG = "/persistent", "/persistent/logs"
PREDICTION_LOG = f"{DIR}/prediction_log.csv"
WRONG = f"{DIR}/wrong_predictions.csv"
EVAL_RESULT = f"{DIR}/evaluation_result.csv"
TRAIN_LOG = f"{LOG}/train_log.csv"
AUDIT_LOG = f"{LOG}/evaluation_audit.csv"
os.makedirs(LOG, exist_ok=True)

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
model_success_tracker = {}

def update_model_success(s, t, m, success):
    k = (s, t or "알수없음", m)
    model_success_tracker.setdefault(k, {"success":0,"fail":0})
    model_success_tracker[k]["success" if success else "fail"] += 1

def get_model_success_rate(s, t, m, min_total=10):
    r = model_success_tracker.get((s, t or "알수없음", m), {"success":0,"fail":0})
    total = r["success"] + r["fail"]
    return 0.5 if total < min_total else r["success"] / total

def load_failure_count():
    path = "/persistent/logs/failure_count.csv"
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}
    except: return {}

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
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0: w.writeheader()
            w.writerow(row)
    except: pass

def log_prediction(symbol, strategy, direction=None, entry_price=0, target_price=0,
                   timestamp=None, model=None, success=True, reason="", rate=0.0,
                   return_value=None, volatility=False, source="일반", predicted_class=None):
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
        "source": source,
        "predicted_class": int(predicted_class) if predicted_class is not None else -1
    }
    log_audit(row["symbol"], row["strategy"], "예측성공" if success else "예측실패", row["reason"])
    try:
        with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0: w.writeheader()
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

def get_dynamic_eval_wait(strategy):
    return {"단기":4, "중기":24, "장기":168}.get(strategy, 6)

def get_feature_hash(feature_row):
    rounded = [round(float(x), 2) for x in feature_row]
    joined = ",".join(map(str, rounded))
    return hashlib.sha1(joined.encode()).hexdigest()

def evaluate_predictions(get_price_fn):
    from failure_db import ensure_failure_db, insert_failure_record, load_existing_failure_hashes, analyze_failure_reason
    from logger import update_model_success, get_feature_hash
    from data.utils import compute_features

    ensure_failure_db()

    if not os.path.exists(PREDICTION_LOG):
        return

    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
        if not rows:
            print("[스킵] 예측 결과 없음 → 평가 건너뜀")
            return
    except Exception as e:
        print(f"[예측 평가 로드 오류] {e}")
        return

    now = now_kst()
    updated, evaluated = [], []

    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}

    # ✅ 중앙값 기준 제거, 하한 도달 이상이면 성공
    class_bins = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]

    for r in rows:
        try:
            if r.get("status") not in ["pending", "failed", "v_pending", "v_failed"]:
                updated.append(r)
                continue

            s = r.get("symbol")
            strat = r.get("strategy")
            m = r.get("model", "unknown")
            entry = float(r.get("entry_price", 0))

            try:
                pred_class = int(r.get("predicted_class", -1))
            except:
                pred_class = -1

            pred_time = datetime.datetime.fromisoformat(r["timestamp"]).astimezone(pytz.timezone("Asia/Seoul"))
            eval_deadline = pred_time + datetime.timedelta(hours=eval_horizon_map.get(strat, 6))
            vol = str(r.get("volatility", "False")).lower() in ["1", "true", "yes"]

            df = get_price_fn(s, strat)
            if df is None or df.empty or "timestamp" not in df.columns:
                r.update({"status": "skip_eval", "reason": "가격 데이터 누락", "return": 0.0})
                updated.append(r)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")

            if now < eval_deadline:
                r.update({
                    "reason": f"⏳ 평가 대기 중 ({now.strftime('%H:%M')} < {eval_deadline.strftime('%H:%M')})",
                    "return": 0.0
                })
            elif entry == 0 or m == "unknown":
                r.update({
                    "status": "v_invalid_model" if vol else "invalid_model",
                    "reason": "모델 없음 또는 entry=0",
                    "return": 0.0
                })
            else:
                eval_df = df[(df["timestamp"] >= pred_time) & (df["timestamp"] <= eval_deadline)]
                if eval_df.empty:
                    r.update({
                        "status": "skip_eval",
                        "reason": "해당 구간 가격 없음",
                        "return": 0.0
                    })
                else:
                    actual_max = eval_df["high"].max()
                    actual_gain = (actual_max - entry) / entry if entry > 0 else 0.0

                    # ✅ 예측 클래스 구간 하한 이상 도달하면 성공
                    if 0 <= pred_class < len(class_bins):
                        threshold = class_bins[pred_class]
                        success = actual_gain >= threshold
                    else:
                        success = False

                    r.update({
                        "status": "v_success" if vol and success else "v_fail" if vol else "success" if success else "fail",
                        "reason": f"예측클래스={pred_class} / 하한={threshold:.3f} / 수익률={actual_gain:.4f}",
                        "return": round(actual_gain, 5)
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

    if evaluated:
        with open(EVAL_RESULT, "a", newline="", encoding="utf-8-sig") as ef:
            w = csv.DictWriter(ef, fieldnames=evaluated[0].keys())
            if os.stat(EVAL_RESULT).st_size == 0:
                w.writeheader()
            w.writerows(evaluated)

        failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
        if failed:
            with open(WRONG, "a", newline="", encoding="utf-8-sig") as wf:
                w = csv.DictWriter(wf, fieldnames=failed[0].keys())
                if os.stat(WRONG).st_size == 0:
                    w.writeheader()
                w.writerows(failed)

            try:
                existing_hashes = load_existing_failure_hashes()
                for r in failed:
                    symbol, strategy = r["symbol"], r["strategy"]
                    if not symbol or not strategy:
                        continue

                    df = get_price_fn(symbol, strategy)
                    df_feat = compute_features(symbol, df, strategy)
                    if df_feat is None or df_feat.empty or "timestamp" not in df_feat.columns:
                        continue

                    df_feat = df_feat.dropna()
                    feature_row = df_feat.drop(columns=["timestamp"]).iloc[-1].values
                    if not isinstance(feature_row, (np.ndarray, list)) or len(feature_row) == 0:
                        continue

                    hash_value = get_feature_hash(feature_row)
                    key = (symbol, strategy, r.get("direction", "예측실패"), hash_value)
                    if key in existing_hashes:
                        continue

                    failure_reason = analyze_failure_reason(
                        float(r.get("rate", 0.0)),
                        df_feat["volatility"].iloc[-1] if "volatility" in df_feat.columns else None
                    )

                    r["reason"] = failure_reason
                    insert_failure_record(r, hash_value)
                    existing_hashes.add(key)
            except Exception as e:
                print(f"[실패패턴 기록 오류] {e}")

    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=updated[0].keys())
        w.writeheader()
        w.writerows(updated)

strategy_stats = {}

# 📁 logger.py 파일 하단에 추가하세요

import pandas as pd
from collections import defaultdict

PREDICTION_LOG = "/persistent/prediction_log.csv"

def analyze_class_success():
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["status"].isin(["success", "fail"])]
        df = df[df["predicted_class"] >= 0]

        result = defaultdict(lambda: {"success": 0, "fail": 0})

        for _, row in df.iterrows():
            strategy = row["strategy"]
            cls = int(row["predicted_class"])
            key = (strategy, cls)
            result[key]["success" if row["status"] == "success" else "fail"] += 1

        summary = []
        for (strategy, cls), counts in result.items():
            total = counts["success"] + counts["fail"]
            rate = counts["success"] / total if total > 0 else 0
            summary.append({
                "strategy": strategy,
                "class": cls,
                "total": total,
                "success": counts["success"],
                "fail": counts["fail"],
                "success_rate": round(rate, 4)
            })

        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values(by=["strategy", "class"])
        return summary_df

    except Exception as e:
        print(f"[오류] 클래스 성공률 분석 실패 → {e}")
        return pd.DataFrame([])
        
def get_fine_tune_targets(min_samples=30, max_success_rate=0.4):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["status"].isin(["success", "fail"])]
        df = df[df["predicted_class"] >= 0]

        result = defaultdict(lambda: {"success": 0, "fail": 0})

        for _, row in df.iterrows():
            strategy = row["strategy"]
            cls = int(row["predicted_class"])
            key = (strategy, cls)
            result[key]["success" if row["status"] == "success" else "fail"] += 1

        fine_tune_targets = []
        for (strategy, cls), counts in result.items():
            total = counts["success"] + counts["fail"]
            rate = counts["success"] / total if total > 0 else 0
            if total < min_samples or rate < max_success_rate:
                fine_tune_targets.append({
                    "strategy": strategy,
                    "class": cls,
                    "samples": total,
                    "success_rate": round(rate, 4)
                })

        return pd.DataFrame(fine_tune_targets).sort_values(by=["strategy", "class"])

    except Exception as e:
        print(f"[오류] fine-tune 대상 분석 실패 → {e}")
        return pd.DataFrame([])
