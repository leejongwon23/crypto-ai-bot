# logger.py
import os, csv, datetime, pandas as pd, pytz, hashlib
import sqlite3
from collections import defaultdict

# -------------------------
# 기본 경로/디렉토리
# -------------------------
DIR = "/persistent"
LOG_DIR = os.path.join(DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ prediction_log는 "루트" 경로로 통일
PREDICTION_LOG = f"{DIR}/prediction_log.csv"
WRONG = f"{DIR}/wrong_predictions.csv"  # (호환 목적으로 유지만)
EVAL_RESULT = f"{LOG_DIR}/evaluation_result.csv"

# ✅ 학습 로그 파일명 통일
TRAIN_LOG = f"{LOG_DIR}/train_log.csv"
AUDIT_LOG = f"{LOG_DIR}/evaluation_audit.csv"

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# -------------------------
# SQLite: 모델 성공/실패 집계
# -------------------------
_db_conn = None
def get_db_connection():
    """lazy sqlite connection (logs/failure_patterns.db)"""
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
    """model_success 테이블 보장"""
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
    """모델별 성공/실패 누적"""
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
    """성공률 없으면 0.0 (참고용)"""
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

# 서버 시작 시 테이블 보장
ensure_success_db()

# -------------------------
# 파일 로드/유틸
# -------------------------
def load_failure_count():
    path = os.path.join(LOG_DIR, "failure_count.csv")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}
    except:
        return {}

def _normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    로그 호환:
    - 새 포맷: 'success' (True/False)
    - 구 포맷: 'status' ('success'/'fail')
    둘 다 지원하도록 df['status']를 생성해 반환.
    """
    if "status" in df.columns:
        df["status"] = (
            df["status"].astype(str).str.lower().map(lambda x: "success" if x == "success" else "fail")
        )
        return df

    if "success" in df.columns:
        s = df["success"]
        s_norm = s.map(lambda x: str(x).strip().lower() in ["true", "1", "yes", "y"])
        df["status"] = s_norm.map(lambda b: "success" if b else "fail")
        return df

    # 둘 다 없으면 빈 status 추가(집계 결과는 0건 처리)
    df["status"] = ""
    return df

def get_actual_success_rate(strategy, min_samples: int = 1):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["strategy"] == strategy]
        df = _normalize_status(df)
        df = df[df["status"].isin(["success", "fail"])]
        n = len(df)
        if n < max(1, min_samples):
            return 0.0
        return round(len(df[df["status"] == "success"]) / n, 4)
    except Exception as e:
        print(f"[오류] get_actual_success_rate 실패 → {e}")
        return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = _normalize_status(df)
        return len(df[(df["strategy"] == strategy) & (df["status"].isin(["success", "fail"]))])
    except Exception as e:
        print(f"[오류] get_strategy_eval_count 실패 → {e}")
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
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
    except:
        pass

# -------------------------
# 예측 로그 기록
# -------------------------
def log_prediction(
    symbol, strategy, direction=None, entry_price=0, target_price=0,
    timestamp=None, model=None, predicted_class=None, top_k=None,
    note="", success=False, reason="", rate=None, return_value=None,
    label=None, group_id=None, model_symbol=None, model_name=None,
    source="일반", volatility=False, feature_vector=None,
    source_exchange="BYBIT"
):
    """
    예측 로그 기록 함수 (표준 경로/헤더 사용)
    source_exchange: BYBIT / BINANCE / MIXED
    """
    from datetime import datetime as _dt
    from failure_db import insert_failure_record  # 외부 모듈 의존

    LOG_FILE = PREDICTION_LOG  # ✅ 루트 경로로 통일
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    now = _dt.now(pytz.timezone("Asia/Seoul")).isoformat() if timestamp is None else timestamp
    top_k_str = ",".join(map(str, top_k)) if top_k else ""

    predicted_class = predicted_class if predicted_class is not None else -1
    label = label if label is not None else -1
    reason = reason or "사유없음"
    rate = 0.0 if rate is None else float(rate)
    return_value = 0.0 if return_value is None else float(return_value)
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
            try:
                import numpy as _np
                if feature_vector is not None:
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
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 6)

def get_feature_hash(feature_row):
    rounded = [round(float(x), 2) for x in feature_row]
    joined = ",".join(map(str, rounded))
    return hashlib.sha1(joined.encode()).hexdigest()

def analyze_class_success(min_samples_per_class: int = 20):
    """
    클래스별 성공/실패 요약.
    - status 컬럼 자동 정규화
    - 샘플 수가 min_samples_per_class 미만이면 success_rate를 None으로 두고 'insufficient'=True
    """
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = _normalize_status(df)
        df = df[df["status"].isin(["success", "fail"])]
        df = df[df["predicted_class"].fillna(-1).astype(int) >= 0]

        result = defaultdict(lambda: {"success": 0, "fail": 0})
        for _, row in df.iterrows():
            key = (row.get("strategy", "알수없음"), int(row["predicted_class"]))
            result[key]["success" if row["status"] == "success" else "fail"] += 1

        summary = []
        for (strategy, cls), cnt in result.items():
            total = cnt["success"] + cnt["fail"]
            if total == 0:
                sr = 0.0
                insufficient = True
            else:
                sr = cnt["success"] / total
                insufficient = total < min_samples_per_class

            summary.append({
                "strategy": strategy,
                "class": cls,
                "total": total,
                "success": cnt["success"],
                "fail": cnt["fail"],
                "success_rate": None if insufficient else round(sr, 4),
                "insufficient": insufficient
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

def get_fine_tune_targets(min_samples: int = 30, max_success_rate: float = 0.4, min_samples_per_class: int = 20):
    """
    파인튜닝 후보 선정:
    - status 정규화
    - (strategy, class) total < min_samples_per_class → 데이터 부족으로 우선 후보
    - total ≥ min_samples_per_class 인 경우에만 성공률 기준 적용
    """
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = _normalize_status(df)
        df = df[df["status"].isin(["success", "fail"])]
        df["predicted_class"] = df["predicted_class"].fillna(-1).astype(int)
        df["label"] = df["label"].fillna(-1).astype(int)
        if "strategy" not in df.columns:
            df["strategy"] = "알수없음"

        result = defaultdict(lambda: {"success": 0, "fail": 0})
        for _, row in df.iterrows():
            key = (row["strategy"], int(row["predicted_class"]))
            result[key]["success" if row["status"] == "success" else "fail"] += 1

        fine_tune_targets = []
        for (strategy, cls), counts in result.items():
            total = counts["success"] + counts["fail"]
            if total < min_samples_per_class:
                fine_tune_targets.append({
                    "strategy": strategy, "class": cls,
                    "samples": total, "success_rate": None, "reason": "insufficient_samples"
                })
                continue

            rate = counts["success"] / total if total > 0 else 0.0
            if counts["fail"] >= 1 or rate < max_success_rate:
                fine_tune_targets.append({
                    "strategy": strategy, "class": cls,
                    "samples": total, "success_rate": round(rate, 4), "reason": "low_perf"
                })

        if len(fine_tune_targets) == 0:
            return pd.DataFrame([])
        df_out = pd.DataFrame(fine_tune_targets).sort_values(by=["strategy", "class"])
        return df_out if len(df_out) >= min_samples else df_out
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
# 사용가능 모델 스캔 (파일명 패턴 고정)
# -------------------------
def get_available_models(symbol: str):
    """
    모델 디렉토리에서 해당 symbol의 (lstm|cnn_lstm|transformer) 모델을 모두 수집
    파일명 예: {symbol}_{strategy}_{model_type}_group{gid}_cls{N}.pt  (접미사 옵션)
    반환: [{"pt_file": "...pt", "model_type":"lstm"}, ...]
    """
    import re
    MODEL_DIR = "/persistent/models"
    result = []
    try:
        pat = re.compile(rf"^{re.escape(symbol)}_(단기|중기|장기)_(lstm|cnn_lstm|transformer)(?:_.*)?\.pt$")
        for f in os.listdir(MODEL_DIR):
            if not f.endswith(".pt"):
                continue
            m = pat.match(f)
            if not m:
                continue
            model_type = m.group(2)  # lstm | cnn_lstm | transformer
            result.append({"pt_file": f, "model_type": model_type})
    except Exception as e:
        print(f"[get_available_models 오류] {e}")
    return result
