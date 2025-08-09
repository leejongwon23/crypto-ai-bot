import os, csv, datetime, pandas as pd, pytz, hashlib
import sqlite3
from collections import defaultdict

DIR = "/persistent"
LOG_DIR = os.path.join(DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ prediction_log는 "루트"로 통일
PREDICTION_LOG = f"{DIR}/prediction_log.csv"
WRONG = f"{DIR}/wrong_predictions.csv"
EVAL_RESULT = f"{LOG_DIR}/evaluation_result.csv"

# ✅ 학습 로그 파일명 통일: train_log.csv
TRAIN_LOG = f"{LOG_DIR}/train_log.csv"
AUDIT_LOG = f"{LOG_DIR}/evaluation_audit.csv"

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
model_success_tracker = {}

# -------------------------
# DB 연결/보조
# -------------------------
_db_conn = None
def get_db_connection():
    global _db_conn
    if _db_conn is None:
        try:
            _db_conn = sqlite3.connect(os.path.join(LOG_DIR, "failure_patterns.db"), check_same_thread=False)
            print("[✅ logger.py DB connection 생성 완료]")
        except Exception as e:
            print(f"[오류] logger.py DB connection 생성 실패 → {e}")
            _db_conn = None
    return _db_conn

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

def get_model_success_rate(s, t, m):
    """성공률 없으면 0.0 반환(차단용 아님, 참고용)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT success, fail FROM model_success
            WHERE symbol=? AND strategy=? AND model=?
        """, (s, t or "알수없음", m))
        row = cur.fetchone()
        if row is None:
            return 0.0
        success_cnt, fail_cnt = row
        total = success_cnt + fail_cnt
        return (success_cnt / total) if total > 0 else 0.0
    except Exception as e:
        print(f"[오류] get_model_success_rate 실패 → {e}")
        return 0.0

# ✅ 서버 시작 시 테이블 보장
ensure_success_db()

def load_failure_count():
    path = os.path.join(LOG_DIR, "failure_count.csv")
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}
    except:
        return {}

def get_actual_success_rate(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        df = df[df["status"].isin(["success", "fail"])]
        return round(len(df[df["status"] == "success"]) / len(df), 4) if len(df) > 0 else 0.0
    except:
        return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return len(df[(df["strategy"] == strategy) & df["status"].isin(["success", "fail"])])
    except:
        return 0

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
    except:
        pass

# -------------------------
# 예측 로그 기록
# -------------------------
def log_prediction(symbol, strategy, direction=None, entry_price=0, target_price=0,
                   timestamp=None, model=None, predicted_class=None, top_k=None,
                   note="", success=False, reason="", rate=None, return_value=None,
                   label=None, group_id=None, model_symbol=None, model_name=None,
                   source="일반", volatility=False, feature_vector=None,
                   source_exchange="BYBIT"):
    """
    예측 로그 기록 함수 (표준 경로/헤더 사용)
    source_exchange: BYBIT / BINANCE / MIXED
    """
    import numpy as np
    from datetime import datetime as _dt
    from failure_db import insert_failure_record

    LOG_FILE = PREDICTION_LOG  # ✅ 루트 경로로 통일
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    now = _dt.now(pytz.timezone("Asia/Seoul")).isoformat() if timestamp is None else timestamp
    top_k_str = ",".join(map(str, top_k)) if top_k else ""

    predicted_class = predicted_class if predicted_class is not None else -1
    label = label if label is not None else -1
    reason = reason or "사유없음"
    rate = 0.0 if rate is None else rate
    return_value = 0.0 if return_value is None else return_value
    entry_price = entry_price or 0.0
    target_price = target_price or 0.0

    allowed_sources = ["일반", "meta", "evo_meta", "baseline_meta", "진화형", "평가", "단일", "변동성", "train_loop"]
    if source not in allowed_sources:
        source = "일반"

    row = [
        now, symbol, strategy, direction, entry_price, target_price,
        (model or ""), predicted_class, top_k_str, note,
        str(success), reason, rate, return_value, label,
        group_id, model_symbol, model_name, source, volatility, source_exchange
    ]

    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "symbol", "strategy", "direction",
                    "entry_price", "target_price",
                    "model", "predicted_class", "top_k", "note",
                    "success", "reason", "rate", "return_value",
                    "label", "group_id", "model_symbol", "model_name",
                    "source", "volatility", "source_exchange"
                ])
            writer.writerow(row)

        print(f"[✅ 예측 로그 기록됨] {symbol}-{strategy} class={predicted_class} | success={success} | src={source_exchange} | reason={reason}")

        # 실패 케이스는 실패 DB에도 기록(중복 체크는 failure_db에서)
        if not success:
            feature_hash = f"{symbol}-{strategy}-{model or ''}-{predicted_class}-{label}-{rate}"
            safe_vector = []
            if feature_vector is not None:
                try:
                    import numpy as _np
                    if isinstance(feature_vector, _np.ndarray):
                        safe_vector = feature_vector.flatten().tolist()
                    elif isinstance(feature_vector, list):
                        safe_vector = feature_vector
                except:
                    safe_vector = []

            insert_failure_record(
                {
                    "symbol": symbol, "strategy": strategy, "direction": direction,
                    "model": model or "", "predicted_class": predicted_class,
                    "rate": rate, "reason": reason, "label": label, "source": source,
                    "entry_price": entry_price, "target_price": target_price,
                    "return_value": return_value
                },
                feature_hash=feature_hash, label=label, feature_vector=safe_vector
            )

    except Exception as e:
        print(f"[⚠️ 예측 로그 기록 실패] {e}")

# -------------------------
# 기타 유틸
# -------------------------
def get_dynamic_eval_wait(strategy):
    return {"단기":4, "중기":24, "장기":168}.get(strategy, 6)

def get_feature_hash(feature_row):
    rounded = [round(float(x), 2) for x in feature_row]
    joined = ",".join(map(str, rounded))
    return hashlib.sha1(joined.encode()).hexdigest()

def analyze_class_success():
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["status"].isin(["success", "fail"])]
        df = df[df["predicted_class"] >= 0]
        result = defaultdict(lambda: {"success": 0, "fail": 0})
        for _, row in df.iterrows():
            key = (row["strategy"], int(row["predicted_class"]))
            result[key]["success" if row["status"] == "success" else "fail"] += 1
        summary = []
        for (strategy, cls), cnt in result.items():
            total = cnt["success"] + cnt["fail"]
            rate = cnt["success"] / total if total > 0 else 0
            summary.append({
                "strategy": strategy, "class": cls, "total": total,
                "success": cnt["success"], "fail": cnt["fail"],
                "success_rate": round(rate, 4)
            })
        return pd.DataFrame(summary).sort_values(by=["strategy", "class"])
    except Exception as e:
        print(f"[오류] 클래스 성공률 분석 실패 → {e}")
        return pd.DataFrame([])

def get_recent_predicted_classes(strategy: str, recent_days: int = 3):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]
        return set(df["predicted_class"].dropna().astype(int).tolist())
    except:
        return set()

def get_fine_tune_targets(min_samples=30, max_success_rate=0.4):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["status"].isin(["success", "fail"])]
        df = df[(df["predicted_class"] >= 0) & (df["label"] >= 0)]
        if "strategy" not in df.columns:
            df["strategy"] = "알수없음"

        result = defaultdict(lambda: {"success": 0, "fail": 0})
        for _, row in df.iterrows():
            key = (row["strategy"], int(row["predicted_class"]))
            result[key]["success" if row["status"] == "success" else "fail"] += 1

        fine_tune_targets = []
        for (strategy, cls), counts in result.items():
            total = counts["success"] + counts["fail"]
            rate = counts["success"] / total if total > 0 else 0
            if counts["fail"] >= 1 or rate < max_success_rate:
                fine_tune_targets.append({
                    "strategy": strategy, "class": cls,
                    "samples": total, "success_rate": round(rate, 4)
                })
        if len(fine_tune_targets) < min_samples:
            return pd.DataFrame(fine_tune_targets)
        return pd.DataFrame(fine_tune_targets).sort_values(by=["strategy", "class"])
    except Exception as e:
        print(f"[오류] fine-tune 대상 분석 실패 → {e}")
        return pd.DataFrame([])

# -------------------------
# prediction_log.csv 존재 보장(헤더 통일)
# -------------------------
PREDICTION_LOG_PATH = PREDICTION_LOG
PREDICTION_HEADERS = [
    "timestamp", "symbol", "strategy", "direction",
    "entry_price", "target_price",
    "model", "predicted_class", "top_k", "note",
    "success", "reason", "rate", "return_value",
    "label", "group_id", "model_symbol", "model_name",
    "source", "volatility", "source_exchange"
]

def ensure_prediction_log_exists():
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    if not os.path.exists(PREDICTION_LOG_PATH):
        pd.DataFrame(columns=PREDICTION_HEADERS).to_csv(
            PREDICTION_LOG_PATH, index=False, encoding="utf-8-sig"
        )
        print("✅ prediction_log.csv 생성(통일 헤더 포함)")
    else:
        print("✅ prediction_log.csv 이미 존재")

# -------------------------
# ✅ 누락되었던 get_available_models 구현
# -------------------------
def get_available_models(symbol: str):
    """
    모델 디렉토리에서 해당 symbol의 (lstm|cnn_lstm|transformer) 모델을 모두 수집
    반환 형식: [{"pt_file": "...pt", "model_type":"lstm"}, ...]
    """
    MODEL_DIR = "/persistent/models"
    result = []
    try:
        for f in os.listdir(MODEL_DIR):
            if not f.startswith(f"{symbol}_"):
                continue
            if not f.endswith(".pt"):
                continue
            parts = f.split("_")
            # expected: symbol_strategy_model[...].pt
            if len(parts) < 3:
                continue
            model_type = parts[2].split(".")[0]
            if model_type not in ["lstm", "cnn_lstm", "transformer"]:
                # 허용: 접미사 포함(ex: transformer_group0_cls20.pt)
                if any(mt in model_type for mt in ["lstm", "cnn_lstm", "transformer"]):
                    # 추출
                    for mt in ["cnn_lstm", "transformer", "lstm"]:
                        if mt in model_type:
                            model_type = mt
                            break
                else:
                    continue
            result.append({"pt_file": f, "model_type": model_type})
    except Exception as e:
        print(f"[get_available_models 오류] {e}")
    return result

# -------------------------
# 학습 로그 기록 (파일명 통일)
# -------------------------
def log_training_result(symbol, strategy, model="", accuracy=0.0, f1=0.0, loss=0.0,
                        note="", source_exchange="BYBIT", status="success"):
    LOG_FILE = TRAIN_LOG  # ✅ train_log.csv 로 통일
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
    row = [now, symbol, strategy, model, accuracy, f1, loss, note, source_exchange, status]
    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "symbol", "strategy", "model",
                    "accuracy", "f1", "loss", "note", "source_exchange", "status"
                ])
            writer.writerow(row)
        print(f"[✅ 학습 로그 기록됨] {symbol}-{strategy} status={status} acc={accuracy:.3f} f1={f1:.3f} src={source_exchange}")
    except Exception as e:
        print(f"[⚠️ 학습 로그 기록 실패] {e}")
