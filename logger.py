import os, csv, datetime, hashlib, sqlite3
import pandas as pd
import pytz
from typing import Dict, Any

# =========================
# 경로/상수
# =========================
DIR = "/persistent"
LOG_DIR = "/persistent/logs"
os.makedirs(LOG_DIR, exist_ok=True)

PREDICTION_LOG_PATH = f"{LOG_DIR}/prediction_log.csv"
WRONG = f"{DIR}/wrong_predictions.csv"
EVAL_RESULT = f"{DIR}/evaluation_result.csv"
TRAIN_LOG = f"{LOG_DIR}/train_log.csv"
AUDIT_LOG = f"{LOG_DIR}/evaluation_audit.csv"
MODEL_DIR = "/persistent/models"

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# =========================
# DB 연결
# =========================
_db_conn = None
def get_db_connection():
    global _db_conn
    if _db_conn is None:
        try:
            os.makedirs(os.path.dirname(f"{LOG_DIR}/failure_patterns.db"), exist_ok=True)
            _db_conn = sqlite3.connect(f"{LOG_DIR}/failure_patterns.db", check_same_thread=False)
            print("[✅ logger] failure_patterns DB 연결 완료")
        except Exception as e:
            print(f"[❌ logger] DB 연결 실패: {e}")
            _db_conn = None
    return _db_conn

# model_success 테이블 보장
def ensure_success_db():
    try:
        conn = get_db_connection()
        if conn is None: return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_success (
                symbol TEXT,
                strategy TEXT,
                model TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                PRIMARY KEY(symbol, strategy, model)
            )
        """)
        conn.commit()
        print("[✅ ensure_success_db] model_success 테이블 확인 완료")
    except Exception as e:
        print(f"[오류] ensure_success_db 실패 → {e}")

ensure_success_db()

# =========================
# 유틸
# =========================
def get_feature_hash(feature_row):
    rounded = [round(float(x), 2) for x in feature_row]
    joined = ",".join(map(str, rounded))
    return hashlib.sha1(joined.encode()).hexdigest()

def get_feature_hash_from_tensor(tensor):
    try:
        flat = tensor.detach().cpu().numpy().flatten()
        rounded = [round(float(x), 2) for x in flat]
        joined = ",".join(map(str, rounded))
        return hashlib.sha1(joined.encode()).hexdigest()
    except Exception as e:
        print(f"[오류] get_feature_hash_from_tensor 실패 → {e}")
        return "unknown"

def log_audit_prediction(s, t, status, reason):
    row = {
        "timestamp": now_kst().isoformat(),
        "symbol": str(s or "UNKNOWN"),
        "strategy": str(t or "알수없음"),
        "status": str(status),
        "reason": str(reason)
    }
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0: w.writeheader()
            w.writerow(row)
    except: pass

# =========================
# 모델 성공 카운터
# =========================
import threading
db_lock = threading.Lock()

def update_model_success(symbol, strategy, model, is_success: bool):
    try:
        conn = get_db_connection()
        if conn is None:
            return
        with db_lock:
            conn.execute("""
                INSERT INTO model_success(symbol,strategy,model,success_count,fail_count)
                VALUES(?,?,?,?,?)
                ON CONFLICT(symbol,strategy,model) DO UPDATE SET
                    success_count = success_count + excluded.success_count,
                    fail_count    = fail_count    + excluded.fail_count
            """, (symbol, strategy or "알수없음", model, 1 if is_success else 0, 0 if is_success else 1))
            conn.commit()
    except Exception as e:
        print(f"[⚠️ update_model_success 실패] {e}")

# =========================
# 예측 로그 보장(헤더 일치)
# =========================
def ensure_prediction_log_exists():
    if not os.path.exists(PREDICTION_LOG_PATH):
        cols = [
            "timestamp","symbol","strategy","direction","entry_price","target_price",
            "model","predicted_class","top_k","note","success","reason",
            "rate","return_value","label","group_id","model_symbol","model_name",
            "source","volatility","source_exchange","status"  # status는 평가에서 사용할 수 있어 추가
        ]
        pd.DataFrame(columns=cols).to_csv(PREDICTION_LOG_PATH, index=False, encoding="utf-8-sig")
        print("✅ prediction_log.csv 생성 완료")
    else:
        print("✅ prediction_log.csv 이미 존재")

ensure_prediction_log_exists()

# =========================
# 모델 검색 (predict.py에서 기대하는 키 형식으로)
# =========================
def get_available_models(target_symbol=None):
    """
    반환 형식 예:
    {
      "model_path": "/persistent/models/BTCUSDT_단기_lstm.pt",
      "symbol": "BTCUSDT",
      "strategy": "단기",
      "model": "lstm",
      "group_id": 0,           # 메타에서 쓰일 수 있음(없으면 None)
      "num_classes": 20,       # 메타에서 쓰일 수 있음(없으면 None)
      "input_size": 24,        # 필수
      "model_name": "BTCUSDT_단기_lstm.pt"
    }
    """
    import json, glob, re
    models = []

    pt_files = glob.glob(os.path.join(MODEL_DIR, "*.pt"))
    for pt_path in pt_files:
        meta_path = pt_path.replace(".pt", ".meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # 필수 필드 체크
            if not all(k in meta for k in ["symbol", "strategy", "model", "input_size"]):
                continue

            if target_symbol and meta["symbol"] != target_symbol:
                continue

            item = {
                "model_path": pt_path,
                "symbol": meta["symbol"],
                "strategy": meta["strategy"],
                "model": meta["model"],
                "input_size": meta["input_size"],
                "model_name": os.path.basename(pt_path),
                "group_id": meta.get("group_id"),
                "num_classes": meta.get("num_classes")
            }
            models.append(item)
        except Exception as e:
            print(f"[⚠️ 메타 로드 실패] {meta_path} → {e}")
            continue
    return models

# =========================
# 통계/조회 보조
# =========================
def get_actual_success_rate(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["strategy"] == strategy]
        if "status" not in df.columns: return 0.0
        df = df[df["status"].isin(["success", "fail","v_success","v_fail"])]
        return round((df["status"].isin(["success","v_success"])).mean(), 4) if len(df) > 0 else 0.0
    except: 
        return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig", on_bad_lines="skip")
        if "status" not in df.columns: return 0
        return int(((df["strategy"]==strategy) & df["status"].isin(["success","fail","v_success","v_fail"])).sum())
    except: 
        return 0

def analyze_class_success():
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty or "status" not in df.columns: return pd.DataFrame([])
        df = df[df["status"].isin(["success", "fail","v_success","v_fail"])]
        df = df[df["predicted_class"].fillna(-1).astype(int) >= 0]

        from collections import defaultdict
        result = defaultdict(lambda: {"success": 0, "fail": 0})

        for _, row in df.iterrows():
            strategy = row.get("strategy","")
            cls = int(row.get("predicted_class", -1))
            if cls < 0: continue
            key = (strategy, cls)
            if row["status"] in ["success","v_success"]:
                result[key]["success"] += 1
            else:
                result[key]["fail"] += 1

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

        summary_df = pd.DataFrame(summary).sort_values(by=["strategy", "class"])
        return summary_df

    except Exception as e:
        print(f"[오류] 클래스 성공률 분석 실패 → {e}")
        return pd.DataFrame([])

def get_recent_predicted_classes(strategy: str, recent_days: int = 3):
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig", on_bad_lines="skip")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[(df["strategy"] == strategy) & (df["timestamp"] >= cutoff)]
        return set(df["predicted_class"].dropna().astype(int).tolist())
    except:
        return set()

def export_recent_model_stats(recent_days=3):
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig", on_bad_lines="skip")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[(df["timestamp"] >= cutoff) & (df["status"].isin(["success","fail","v_success","v_fail"]))]

        if df.empty:
            print("❗ 최근 예측 데이터 없음")
            return

        from collections import defaultdict
        stats = defaultdict(lambda: {"success": 0, "fail": 0})
        for _, row in df.iterrows():
            key = (row.get("symbol",""), row.get("strategy",""), row.get("model",""))
            if row["status"] in ["success","v_success"]:
                stats[key]["success"] += 1
            else:
                stats[key]["fail"] += 1

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

        save_path = f"{LOG_DIR}/recent_model_stats.csv"
        pd.DataFrame(summary).to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"📈 최근 모델 성능 저장 완료 → {save_path}")

    except Exception as e:
        print(f"[오류] 최근 모델 성능 집계 실패 → {e}")

# =========================
# 예측 로그 기록 (경계 일치)
# =========================
def log_prediction(
    symbol, strategy, direction=None, entry_price=0, target_price=0,
    timestamp=None, model=None, predicted_class=None, top_k=None,
    note="", success=False, reason="", rate=None, return_value=None,
    label=None, group_id=None, model_symbol=None, model_name=None,
    source="일반", volatility=False, feature_vector=None,
    source_exchange="BYBIT"
):
    """
    예측 로그 기록 함수
    - 성공/실패 판정은 config의 (symbol,strategy)별 클래스 경계 사용
    """
    import json
    import numpy as np
    from config import get_class_return_range
    from failure_db import insert_failure_record

    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat() if timestamp is None else timestamp
    top_k_str = ",".join(map(str, top_k)) if top_k else ""

    # 필드 기본값
    predicted_class = int(predicted_class) if predicted_class is not None else -1
    label = int(label) if label is not None else -1
    reason = reason or "사유없음"
    rate = float(rate) if rate is not None else 0.0
    return_value = float(return_value) if return_value is not None else 0.0
    entry_price = float(entry_price) if entry_price else 0.0
    target_price = float(target_price) if target_price else 0.0

    allowed_sources = ["일반", "meta", "evo_meta", "baseline_meta", "진화형"]
    if source not in allowed_sources:
        source = "일반"

    # ✅ (symbol,strategy) 경계로 성공 판정
    if predicted_class >= 0 and symbol and strategy:
        try:
            cls_min, _ = get_class_return_range(predicted_class, symbol, strategy)
            success = return_value >= cls_min
        except Exception as e:
            print(f"[⚠️ get_class_return_range 오류] {e}")

    row = [
        now, symbol, strategy, direction, entry_price, target_price,
        model or "", predicted_class, top_k_str, note,
        str(bool(success)), reason, rate, return_value, label,
        group_id, model_symbol, model_name, source, volatility, source_exchange
    ]

    try:
        write_header = not os.path.exists(PREDICTION_LOG_PATH)
        with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "symbol", "strategy", "direction", "entry_price", "target_price",
                    "model", "predicted_class", "top_k", "note", "success", "reason",
                    "rate", "return_value", "label", "group_id", "model_symbol", "model_name",
                    "source", "volatility", "source_exchange"
                ])
            writer.writerow(row)

        print(f"[✅ 예측 로그] {symbol}-{strategy} class={predicted_class} | success={success} | src={source_exchange} | reason={reason}")

        # 실패 케이스는 실패 DB 적재
        if not success:
            feature_hash = f"{symbol}-{strategy}-{model or ''}-{predicted_class}-{label}-{rate}"
            if feature_vector is None:
                feature_vector = []
            elif hasattr(feature_vector, "detach"):  # torch.Tensor
                try:
                    feature_vector = feature_vector.detach().cpu().numpy().flatten().tolist()
                except:
                    feature_vector = []
            elif not isinstance(feature_vector, list):
                try:
                    feature_vector = list(feature_vector)
                except:
                    feature_vector = []

            with db_lock:
                insert_failure_record(
                    {
                        "symbol": symbol, "strategy": strategy, "direction": direction,
                        "model": model or "", "predicted_class": predicted_class,
                        "rate": rate, "reason": reason, "label": label, "source": source,
                        "entry_price": entry_price, "target_price": target_price,
                        "return_value": return_value
                    },
                    feature_hash=feature_hash,
                    feature_vector=feature_vector,
                    label=label
                )
    except Exception as e:
        print(f"[❌ log_prediction 실패] {e}")
