import os, csv, datetime, pandas as pd, pytz, hashlib
from data.utils import get_kline_by_strategy
import pandas as pd
import sqlite3

DIR, LOG = "/persistent", "/persistent/logs"
PREDICTION_LOG = f"{DIR}/prediction_log.csv"
WRONG = f"{DIR}/wrong_predictions.csv"
EVAL_RESULT = f"{DIR}/evaluation_result.csv"
TRAIN_LOG = f"{LOG}/train_log.csv"
AUDIT_LOG = f"{LOG}/evaluation_audit.csv"
os.makedirs(LOG, exist_ok=True)

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
model_success_tracker = {}


def get_db_connection():
    import sqlite3
    global _db_conn
    if '_db_conn' not in globals() or _db_conn is None:
        try:
            _db_conn = sqlite3.connect("/persistent/logs/failure_patterns.db", check_same_thread=False)
            print("[✅ logger.py DB connection 생성 완료]")
        except Exception as e:
            print(f"[오류] logger.py DB connection 생성 실패 → {e}")
            _db_conn = None
    return _db_conn


import sqlite3

DB_PATH = "/persistent/logs/failure_patterns.db"

def ensure_success_db():
    try:
        conn = get_db_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_success (
                symbol TEXT,
                strategy TEXT,
                model TEXT,
                success INTEGER,
                fail INTEGER,
                PRIMARY KEY(symbol, strategy, model)
            )
        """)
        conn.commit()
        print("[✅ ensure_success_db] model_success 테이블 확인 완료")
    except Exception as e:
        print(f"[오류] ensure_success_db 실패 → {e}")

def update_model_success(s, t, m, success):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_success (symbol, strategy, model, success, fail)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, strategy, model) DO UPDATE SET
                success = success + excluded.success,
                fail = fail + excluded.fail
        """, (s, t or "알수없음", m, int(success), int(not success)))
        conn.commit()
        print(f"[✅ update_model_success] {s}-{t}-{m} 기록 ({'성공' if success else '실패'})")
    except Exception as e:
        print(f"[오류] update_model_success 실패 → {e}")

def get_model_success_rate(s, t, m, min_total=10):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT success, fail FROM model_success
            WHERE symbol=? AND strategy=? AND model=?
        """, (s, t or "알수없음", m))
        row = cur.fetchone()

        if row is None:
            print(f"[INFO] {s}-{t}-{m}: 기록 없음 → cold-start 0.2 반환")
            return 0.2

        success_cnt, fail_cnt = row
        total = success_cnt + fail_cnt

        if total < min_total:
            fail_ratio = fail_cnt / total if total > 0 else 1.0
            weight = max(0.0, 1.0 - fail_ratio)
            final_weight = min(weight, 0.2)
            print(f"[INFO] {s}-{t}-{m}: 평가 샘플 부족(total={total}) → weight={final_weight:.2f}")
            return final_weight

        rate = success_cnt / total
        return max(0.0, min(rate, 1.0))

    except Exception as e:
        print(f"[오류] get_model_success_rate 실패 → {e}")
        return 0.2


# ✅ 서버 시작 시 호출
ensure_success_db()

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

import threading
db_lock = threading.Lock()  # ✅ Lock 전역 선언

def log_prediction(symbol, strategy, direction=None, entry_price=0, target_price=0,
                   timestamp=None, model=None, predicted_class=None, top_k=None, note=""):
    import csv
    import os
    from datetime import datetime
    import pytz

    LOG_FILE = "/persistent/logs/prediction_log.csv"
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    now = datetime.now(pytz.timezone("Asia/Seoul")).isoformat() if timestamp is None else timestamp
    top_k_str = ",".join(map(str, top_k)) if top_k else ""

    row = [now, symbol, strategy, direction, entry_price, target_price,
           model or "", predicted_class if predicted_class is not None else "", top_k_str, note]

    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "symbol", "strategy", "direction", "entry_price", "target_price",
                                 "model", "predicted_class", "top_k", "note"])
            writer.writerow(row)
        print(f"[✅ 예측 로그 기록됨] {symbol}-{strategy} → class={predicted_class} | top_k={top_k_str}")
    except Exception as e:
        print(f"[⚠️ 예측 로그 기록 실패] {e}")

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

def log_training_result(symbol, strategy, model="", accuracy=0.0, f1=0.0, loss=0.0, note=""):
    import csv
    from datetime import datetime
    import pytz
    import os

    LOG_FILE = "/persistent/logs/training_log.csv"
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    now = datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
    row = [now, symbol, strategy, model, accuracy, f1, loss, note]

    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "symbol", "strategy", "model", "accuracy", "f1", "loss", "note"])
            writer.writerow(row)
        print(f"[✅ 학습 로그 기록됨] {symbol}-{strategy} acc={accuracy:.3f} f1={f1:.3f}")
    except Exception as e:
        print(f"[⚠️ 학습 로그 기록 실패] {e}")

# ✅ 로그 읽기 시 utf-8-sig + 오류 무시
def read_training_log():
    import pandas as pd
    from logger import TRAIN_LOG

    try:
        df = pd.read_csv(TRAIN_LOG, encoding="utf-8-sig", errors="ignore")
        return df
    except Exception as e:
        print(f"[❌ 학습 로그 읽기 오류] {e}")
        return pd.DataFrame()

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

def get_available_models(target_symbol=None):
    import os, json, glob
    from model_weight_loader import get_similar_symbol
    from config import get_SYMBOLS

    MODEL_DIR = "/persistent/models"
    models = []

    # ✅ 전역 SYMBOLS 기준으로 제한
    allowed_symbols = set(get_SYMBOLS())

    # ✅ 유사 symbol 목록
    similar_symbols = []
    if target_symbol:
        similar_symbols = get_similar_symbol(target_symbol)
        similar_symbols.append(target_symbol)  # 자기자신 포함 보장

    pt_files = glob.glob(os.path.join(MODEL_DIR, "*.pt"))
    for pt_path in pt_files:
        meta_path = pt_path.replace(".pt", ".meta.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if not all(k in meta for k in ["symbol", "strategy", "model", "input_size", "model_name"]):
                continue

            # ✅ 허용된 심볼인지
            if meta["symbol"] not in allowed_symbols:
                continue

            # ✅ 유사 심볼이 지정되었을 경우, 자기 자신은 무조건 허용
            if target_symbol and meta["symbol"] != target_symbol and meta["symbol"] not in similar_symbols:
                continue

            model_file = os.path.basename(pt_path)
            models.append({
                "symbol": meta["symbol"],
                "strategy": meta["strategy"],
                "model": meta["model"],
                "pt_file": model_file,
                "group_id": meta.get("group_id"),
                "window": meta.get("window"),
                "input_size": meta["input_size"],
                "model_name": meta.get("model_name", model_file)
            })

        except Exception as e:
            print(f"[⚠️ 메타 로드 실패] {meta_path} → {e}")
            continue

    return models
