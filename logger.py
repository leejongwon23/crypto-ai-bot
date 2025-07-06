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

    if total < min_total:
        # ✅ 평가 샘플 부족 시 실패율 반영 weight 계산
        fail_ratio = r["fail"] / total if total > 0 else 1.0
        weight = max(0.0, 1.0 - fail_ratio)  # 성공률 기반 weight
        cold_start_weight = 0.2
        final_weight = min(weight, cold_start_weight)
        print(f"[INFO] {s}-{t}-{m}: 평가 샘플 부족(total={total}) → weight={final_weight:.2f} (fail_ratio={fail_ratio:.2f})")
        return final_weight

    rate = r["success"] / total
    return max(0.0, min(rate, 1.0))


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

def log_audit_prediction(s, t, status, reason):
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
                   return_value=None, volatility=False, source="일반", predicted_class=None, label=None,
                   augmentation=None, group_id=None):  # ✅ group_id 파라미터 추가

    import csv, os, datetime, pytz

    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    now = timestamp or now_kst().isoformat()
    date_str = now.split("T")[0]
    dated_path = f"/persistent/logs/prediction_{date_str}.csv"
    full_path = "/persistent/prediction_log.csv"

    try:
        pred_class_val = int(float(predicted_class)) if predicted_class not in [None, ""] else -1
    except:
        pred_class_val = -1

    if label is None or str(label).strip() == "":
        label = pred_class_val
    else:
        try:
            label = int(label)
        except:
            label = -1

    if augmentation == "smote":
        status = "smote_aug"
    else:
        status = "v_success" if success and volatility else \
                 "v_fail" if not success and volatility else \
                 "success" if success else "fail"

    effective_rate = rate if rate is not None else 0.0
    effective_return = return_value if return_value is not None else effective_rate

    row = {
        "timestamp": now,
        "symbol": str(symbol or "UNKNOWN"),
        "strategy": str(strategy or "알수없음"),
        "direction": direction or "N/A",
        "entry_price": float(entry_price or 0.0),
        "target_price": float(target_price or 0.0),
        "model": str(model or "unknown"),
        "rate": float(effective_rate),
        "status": status,
        "reason": reason or "",
        "return": float(effective_return),
        "volatility": bool(volatility),
        "source": str(source or "일반"),
        "predicted_class": str(pred_class_val),
        "label": str(label),
        "group_id": str(group_id) if group_id is not None else ""  # ✅ group_id 추가
    }

    row = {str(k): (v if v is not None else "") for k, v in row.items() if k is not None}
    fieldnames = sorted(row.keys())

    for path in [dated_path, full_path]:
        try:
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            print(f"[✅ log_prediction 기록 완료] {path}")
        except Exception as e:
            print(f"[오류] log_prediction 기록 실패 ({path}) → {e}")

def get_dynamic_eval_wait(strategy):
    return {"단기":4, "중기":24, "장기":168}.get(strategy, 6)

def get_feature_hash(feature_row):
    rounded = [round(float(x), 2) for x in feature_row]
    joined = ",".join(map(str, rounded))
    return hashlib.sha1(joined.encode()).hexdigest()

strategy_stats = {}

# 📁 logger.py 파일 하단에 추가하세요

import pandas as pd
from collections import defaultdict

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
    import pandas as pd
    from collections import defaultdict
    import numpy as np

    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["status"].isin(["success", "fail"])]

        # ✅ 라벨 오류 제거
        df = df[(df["predicted_class"] >= 0) & (df["label"] >= 0)]

        if "strategy" not in df.columns:
            df["strategy"] = "알수없음"

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
            # ✅ 실패가 있거나 성공률이 낮으면 fine-tune
            if counts["fail"] >= 1 or rate < max_success_rate:
                fine_tune_targets.append({
                    "strategy": strategy,
                    "class": cls,
                    "samples": total,
                    "success_rate": round(rate, 4)
                })

        # ✅ 최소 min_samples 보장 + 클래스 다양성 확보
        if len(fine_tune_targets) < min_samples:
            print("[INFO] fine-tune 대상 부족 → fallback 최근 실패 + noise sample 사용")
            fail_df = df[df["status"] == "fail"]
            fallback_df = fail_df.sample(n=min_samples, replace=True) if len(fail_df) >= min_samples else fail_df

            fallback = []
            for _, row in fallback_df.iterrows():
                fallback.append({
                    "strategy": row["strategy"],
                    "class": int(row["predicted_class"]),
                    "samples": 10,
                    "success_rate": 0.0
                })

            # ✅ noise sample 추가
            noise_needed = min_samples - len(fallback)
            for i in range(noise_needed):
                fallback.append({
                    "strategy": "noise_aug",
                    "class": np.random.randint(0, 21),
                    "samples": 1,
                    "success_rate": 0.0
                })

            return pd.DataFrame(fallback).sort_values(by=["strategy", "class"])

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
        
def export_recent_model_stats(recent_days=3):
    """
    최근 N일간 모델별 성공률 집계 → CSV 저장
    """
    try:
        path = "/persistent/prediction_log.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # 최근 기간 필터링
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]
        df = df[df["status"].isin(["success", "fail"])]

        if df.empty:
            print("❗ 최근 예측 데이터 없음")
            return

        from collections import defaultdict
        stats = defaultdict(lambda: {"success": 0, "fail": 0})

        for _, row in df.iterrows():
            key = (row["symbol"], row["strategy"], row["model"])
            stats[key]["success" if row["status"] == "success" else "fail"] += 1

        summary = []
        for (symbol, strategy, model), count in stats.items():
            total = count["success"] + count["fail"]
            rate = count["success"] / total if total > 0 else 0
            summary.append({
                "symbol": symbol,
                "strategy": strategy,
                "model": model,
                "total": total,
                "success": count["success"],
                "fail": count["fail"],
                "recent_success_rate": round(rate, 4)
            })

        summary_df = pd.DataFrame(summary)
        save_path = "/persistent/logs/recent_model_stats.csv"
        summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"📈 최근 모델 성능 저장 완료 → {save_path}")

    except Exception as e:
        print(f"[오류] 최근 모델 성능 집계 실패 → {e}")

def log_training_result(symbol, strategy, model_name, acc, f1, loss):
    """
    모델 학습 결과를 로그로 저장
    - 이어 학습 여부가 있는 경우 로그로 명확히 남김
    """
    try:
        import os
        timestamp = now_kst().strftime("%Y-%m-%d %H:%M:%S")
        model_path = f"/persistent/models/{symbol}_{strategy}_{model_name}.pt"
        mode = "이어학습" if os.path.exists(model_path) else "신규학습"

        row = {
            "timestamp": timestamp,
            "symbol": symbol,
            "strategy": strategy,
            "model": model_name,
            "mode": mode,
            "accuracy": float(acc),
            "f1_score": float(f1),
            "loss": float(loss)
        }

        headers = list(row.keys())
        pd.DataFrame([row]).to_csv(TRAIN_LOG, mode="a", index=False,
                                   header=not os.path.exists(TRAIN_LOG),
                                   encoding="utf-8-sig")
    except Exception as e:
        print(f"[학습 로그 저장 오류] {e}")

def get_class_success_rate(strategy, recent_days=3):
    from collections import defaultdict
    import pandas as pd
    import os

    path = "/persistent/prediction_log.csv"
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)

        df = df[(df["strategy"] == strategy) &
                (df["timestamp"] >= cutoff) &
                (df["predicted_class"] >= 0) &
                (df["status"].isin(["success", "fail"]))]

        stats = defaultdict(lambda: {"success": 0, "fail": 0})
        for _, row in df.iterrows():
            cls = int(row["predicted_class"])
            if row["status"] == "success":
                stats[cls]["success"] += 1
            else:
                stats[cls]["fail"] += 1

        result = {}
        for cls, val in stats.items():
            total = val["success"] + val["fail"]
            if total > 0:
                result[cls] = round(val["success"] / total, 4)

        return result

    except Exception as e:
        print(f"[⚠️ 클래스 성공률 계산 오류] {e}")
        return {}



import os

MODEL_DIR = "/persistent/models"


def get_available_models():
    import os, json, glob
    MODEL_DIR = "/persistent/models"

    models = []
    pt_files = glob.glob(os.path.join(MODEL_DIR, "*.pt"))
    for pt_path in pt_files:
        meta_path = pt_path.replace(".pt", ".meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if all(k in meta for k in ["symbol", "strategy", "model", "input_size"]):
            models.append({
                "symbol": meta["symbol"],
                "strategy": meta["strategy"],
                "model": meta["model"],
                "pt_file": os.path.basename(pt_path)
            })
    return models
