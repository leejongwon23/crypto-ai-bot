import os, csv, datetime, pandas as pd, pytz, hashlib
from data.utils import get_kline_by_strategy
import pandas as pd

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
    import csv, datetime, pytz
    import pandas as pd
    from failure_db import ensure_failure_db, insert_failure_record
    from logger import update_model_success
    from collections import defaultdict

    ensure_failure_db()

    PREDICTION_LOG = "/persistent/prediction_log.csv"
    EVAL_RESULT = "/persistent/evaluation_result.csv"
    WRONG = "/persistent/wrong_predictions.csv"
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
        if not rows:
            print("[스킵] 예측 결과 없음 → 평가 건너뜀")
            return
    except Exception as e:
        print(f"[예측 평가 로드 오류] {e}")
        return

    updated, evaluated = [], []
    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}

    # ✅ 클래스 구간 정의 (14개)
    class_ranges = [
        (-0.30, -0.15), (-0.15, -0.10), (-0.10, -0.07), (-0.07, -0.05),
        (-0.05, -0.03), (-0.03, -0.015), (-0.015, -0.01),
        ( 0.01,  0.015), ( 0.015, 0.03), ( 0.03, 0.05), ( 0.05, 0.07),
        ( 0.07, 0.10), ( 0.10, 0.15), ( 0.15, 0.30)
    ]

    for r in rows:
        try:
            if r.get("status") not in ["pending", "failed", "v_pending", "v_failed"]:
                updated.append(r)
                continue

            symbol = r.get("symbol")
            strategy = r.get("strategy")
            model = r.get("model", "unknown")
            entry_price = float(r.get("entry_price", 0))

            # ✅ predicted_class 유효성 확인
            try:
                pred_class = int(r.get("predicted_class", -1))
            except:
                pred_class = -1

            if pred_class == -1:
                r.update({
                    "status": "skip_eval",
                    "reason": "유효하지 않은 클래스",
                    "return": 0.0
                })
                updated.append(r)
                continue

            timestamp = pd.to_datetime(r["timestamp"], utc=True).tz_convert("Asia/Seoul")
            horizon = eval_horizon_map.get(strategy, 6)
            deadline = timestamp + pd.Timedelta(hours=horizon)
            now = now_kst()

            df = get_price_fn(symbol, strategy)
            if df is None or df.empty or "timestamp" not in df.columns:
                r.update({"status": "skip_eval", "reason": "가격 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")

            if now < deadline:
                r.update({"reason": f"⏳ 평가 대기 중 ({now.strftime('%H:%M')} < {deadline.strftime('%H:%M')})", "return": 0.0})
                updated.append(r)
                continue

            future_df = df[(df["timestamp"] >= timestamp) & (df["timestamp"] <= deadline)]
            if future_df.empty:
                r.update({"status": "skip_eval", "reason": "미래 구간 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            actual_max = future_df["high"].max()
            gain = (actual_max - entry_price) / (entry_price + 1e-6)

            if 0 <= pred_class < len(class_ranges):
                low, high = class_ranges[pred_class]
                success = gain >= low
            else:
                success = False
                low, high = 0.0, 0.0

            vol = str(r.get("volatility", "False")).lower() in ["1", "true", "yes"]
            status = "v_success" if vol and success else "v_fail" if vol else "success" if success else "fail"

            r.update({
                "status": status,
                "reason": f"예측={pred_class} / 구간=({low:.3f}~{high:.3f}) / 실현={gain:.4f}",
                "return": round(gain, 5)
            })
            update_model_success(symbol, strategy, model, success)
            evaluated.append(dict(r))

        except Exception as e:
            r.update({"status": "skip_eval", "reason": f"예외 발생: {e}", "return": 0.0})
            updated.append(r)

    if evaluated:
        with open(EVAL_RESULT, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=evaluated[0].keys())
            if os.stat(EVAL_RESULT).st_size == 0:
                w.writeheader()
            w.writerows(evaluated)

        failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
        if failed:
            with open(WRONG, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=failed[0].keys())
                if os.stat(WRONG).st_size == 0:
                    w.writeheader()
                w.writerows(failed)

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

def get_recent_predicted_classes(strategy: str, recent_days: int = 3):
    try:
        df = pd.read_csv("/persistent/prediction_log.csv")
        df = df[df["strategy"] == strategy]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]
        return set(df["predicted_class"].dropna().astype(int).tolist())
    except:
        return set()
        
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

def get_feature_hash_from_tensor(tensor):
    """
    텐서 데이터를 받아 해시값 생성 (학습 피처 중복 방지용)
    """
    try:
        flat = tensor.detach().cpu().numpy().flatten()
        rounded = [round(float(x), 2) for x in flat]
        joined = ",".join(map(str, rounded))
        return hashlib.sha1(joined.encode()).hexdigest()
    except Exception as e:
        print(f"[오류] get_feature_hash_from_tensor 실패 → {e}")
        return "unknown"

