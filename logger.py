# === logger.py (호환 최종본) ===
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
WRONG = f"{DIR}/wrong_predictions.csv"  # (호환 목적)
EVAL_RESULT = f"{LOG_DIR}/evaluation_result.csv"

# ✅ 학습 로그 파일명 통일
TRAIN_LOG = f"{LOG_DIR}/train_log.csv"
AUDIT_LOG = f"{LOG_DIR}/evaluation_audit.csv"

# ✅ 공용 헤더
PREDICTION_HEADERS = [
    "timestamp","symbol","strategy","direction",
    "entry_price","target_price",
    "model","predicted_class","top_k","note",
    "success","reason","rate","return_value",
    "label","group_id","model_symbol","model_name",
    "source","volatility","source_exchange"
]

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# -------------------------
# 안전한 로그 파일 보장
# -------------------------
def ensure_prediction_log_exists():
    try:
        os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)
        if not os.path.exists(PREDICTION_LOG):
            with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(PREDICTION_HEADERS)
            print("[✅ ensure_prediction_log_exists] prediction_log.csv 생성 완료")
        else:
            try:
                with open(PREDICTION_LOG, "r", encoding="utf-8-sig") as f:
                    first_line = f.readline()
                if "," not in first_line or any(h not in first_line for h in ["timestamp","symbol","strategy"]):
                    bak = PREDICTION_LOG + ".bak"
                    os.replace(PREDICTION_LOG, bak)
                    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
                        csv.writer(f).writerow(PREDICTION_HEADERS)
                    with open(bak, "r", encoding="utf-8-sig") as src, open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as dst:
                        dst.write(src.read())
                    print("[✅ ensure_prediction_log_exists] 기존 파일 헤더 보정 완료")
            except Exception as e:
                print(f"[⚠️ ensure_prediction_log_exists] 헤더 확인 실패: {e}")
    except Exception as e:
        print(f"[⚠️ ensure_prediction_log_exists] 예외: {e}")

# -------------------------
# feature hash 유틸
# -------------------------
def get_feature_hash(feature_row) -> str:
    try:
        import numpy as _np
        if feature_row is None:
            return "none"
        if "torch" in str(type(feature_row)):
            try:
                feature_row = feature_row.detach().cpu().numpy()
            except Exception:
                pass
        if isinstance(feature_row, _np.ndarray):
            arr = feature_row.flatten().astype(float)
        elif isinstance(feature_row, (list, tuple)):
            arr = _np.array(feature_row, dtype=float).flatten()
        else:
            arr = _np.array([float(feature_row)], dtype=float)
        rounded = [round(float(x), 2) for x in arr]
        joined = ",".join(map(str, rounded))
        return hashlib.sha1(joined.encode()).hexdigest()
    except Exception:
        return "hash_error"

# -------------------------
# SQLite: 모델 성공/실패 집계
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

# 서버 시작 시 보장
ensure_success_db()
ensure_prediction_log_exists()

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
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower().map(lambda x: "success" if x == "success" else "fail")
        return df
    if "success" in df.columns:
        s = df["success"].map(lambda x: str(x).strip().lower() in ["true","1","yes","y"])
        df["status"] = s.map(lambda b: "success" if b else "fail")
        return df
    df["status"] = ""
    return df

def get_actual_success_rate(strategy, min_samples: int = 1):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["strategy"] == strategy]
        df = _normalize_status(df)
        df = df[df["status"].isin(["success","fail"])]
        n = len(df)
        if n < max(1, min_samples):
            return 0.0
        return round(len(df[df["status"]=="success"]) / n, 4)
    except Exception as e:
        print(f"[오류] get_actual_success_rate 실패 → {e}")
        return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", on_bad_lines="skip")
        df = _normalize_status(df)
        return len(df[(df["strategy"]==strategy) & (df["status"].isin(['success','fail']))])
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
# 예측 로그
# -------------------------
def log_prediction(
    symbol, strategy, direction=None, entry_price=0, target_price=0,
    timestamp=None, model=None, predicted_class=None, top_k=None,
    note="", success=False, reason="", rate=None, return_value=None,
    label=None, group_id=None, model_symbol=None, model_name=None,
    source="일반", volatility=False, feature_vector=None,
    source_exchange="BYBIT"
):
    from datetime import datetime as _dt
    from failure_db import insert_failure_record

    LOG_FILE = PREDICTION_LOG
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

    allowed_sources = ["일반","meta","evo_meta","baseline_meta","진화형","평가","단일","변동성","train_loop"]
    if source not in allowed_sources:
        source = "일반"

    row = [
        now, symbol, strategy, direction, entry_price, target_price,
        (model or ""), predicted_class, top_k_str, note,
        str(success), reason, rate, return_value, label,
        group_id, model_symbol, model_name, source, volatility, source_exchange
    ]

    try:
        write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(PREDICTION_HEADERS)
            writer.writerow(row)

        print(f"[✅ 예측 로그 기록됨] {symbol}-{strategy} class={predicted_class} | success={success} | src={source_exchange} | reason={reason}")

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
# 학습 로그
# -------------------------
def log_training_result(
    symbol, strategy, model="", accuracy=0.0, f1=0.0, loss=0.0,
    note="", source_exchange="BYBIT", status="success",
):
    LOG_FILE = TRAIN_LOG
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
    row = [
        now, str(symbol), str(strategy), str(model or ""),
        float(accuracy) if accuracy is not None else 0.0,
        float(f1) if f1 is not None else 0.0,
        float(loss) if loss is not None else 0.0,
        str(note or ""), str(source_exchange or "BYBIT"),
        str(status or "success")
    ]
    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","model","accuracy","f1","loss","note","source_exchange","status"])
            w.writerow(row)
        print(f"[✅ 학습 로그 기록] {symbol}-{strategy} {model} status={status}")
    except Exception as e:
        print(f"[⚠️ 학습 로그 기록 실패] {e}")
    try:
        ensure_success_db()
        update_model_success(symbol, strategy, model or "", str(status).lower() == "success")
    except Exception as e:
        print(f"[⚠️ model_success 집계 실패] {e}")

# -------------------------
# 수익률 클래스 경계 로그 (호출 호환)
# -------------------------
def log_class_ranges(symbol, strategy, group_id=None, class_ranges=None, note=""):
    """
    /persistent/logs/class_ranges.csv
    컬럼: timestamp,symbol,strategy,group_id,idx,low,high,note
    """
    import csv, os
    path = os.path.join(LOG_DIR, "class_ranges.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = now_kst().isoformat()

    class_ranges = class_ranges or []
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","idx","low","high","note"])
            for i, rng in enumerate(class_ranges):
                try:
                    lo, hi = (float(rng[0]), float(rng[1]))
                except Exception:
                    lo, hi = (None, None)
                w.writerow([now, symbol, strategy, int(group_id) if group_id is not None else 0, i, lo, hi, str(note or "")])
        print(f"[📐 클래스경계 로그] {symbol}-{strategy}-g{group_id} → {len(class_ranges)}개 기록")
    except Exception as e:
        print(f"[⚠️ 클래스경계 로그 실패] {e}")

# -------------------------
# 수익률 분포 요약 로그 (신규)
# -------------------------
def log_return_distribution(symbol, strategy, group_id=None, horizon_hours=None, summary: dict=None, note=""):
    """
    /persistent/logs/return_distribution.csv
    컬럼: timestamp,symbol,strategy,group_id,horizon_hours,min,p25,p50,p75,p90,p95,p99,max,count,note
    """
    path = os.path.join(LOG_DIR, "return_distribution.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = now_kst().isoformat()

    s = summary or {}
    row = [
        now, str(symbol), str(strategy),
        int(group_id) if group_id is not None else 0,
        int(horizon_hours) if horizon_hours is not None else "",
        float(s.get("min", 0.0)), float(s.get("p25", 0.0)), float(s.get("p50", 0.0)),
        float(s.get("p75", 0.0)), float(s.get("p90", 0.0)), float(s.get("p95", 0.0)),
        float(s.get("p99", 0.0)), float(s.get("max", 0.0)), int(s.get("count", 0)),
        str(note or "")
    ]

    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","horizon_hours",
                            "min","p25","p50","p75","p90","p95","p99","max","count","note"])
            w.writerow(row)
        print(f"[📈 수익률분포 로그] {symbol}-{strategy}-g{group_id} count={s.get('count',0)}")
    except Exception as e:
        print(f"[⚠️ 수익률분포 로그 실패] {e}")

# -------------------------
# 라벨 분포 로그 (두 형태 모두 지원)
# -------------------------
def log_label_distribution(
    symbol, strategy, group_id=None,
    counts: dict=None, total: int=None, n_unique: int=None, entropy: float=None,
    labels=None, note=""
):
    """
    호출 호환:
      1) train.py 최신: counts=..., total=..., n_unique=..., entropy=...
      2) 구버전: labels=[...]
    기록: /persistent/logs/label_distribution.csv
    """
    import json, math

    path = os.path.join(LOG_DIR, "label_distribution.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = now_kst().isoformat()

    if counts is None:
        # labels 기반으로 계산
        from collections import Counter
        try:
            labels_list = list(map(int, list(labels or [])))
        except Exception:
            labels_list = []
        cnt = Counter(labels_list)
        total_calc = sum(cnt.values())
        probs = [c/total_calc for c in cnt.values()] if total_calc > 0 else []
        entropy_calc = -sum(p*math.log(p + 1e-12) for p in probs) if probs else 0.0
        counts = {int(k): int(v) for k, v in sorted(cnt.items())}
        total = total_calc
        n_unique = len(cnt)
        entropy = round(float(entropy_calc), 6)
    else:
        # counts 기반(이미 계산된 값 사용)
        counts = {int(k): int(v) for k, v in sorted(counts.items())}
        total = int(total if total is not None else sum(counts.values()))
        n_unique = int(n_unique if n_unique is not None else len(counts))
        if entropy is None:
            # 안전 계산
            import math
            probs = [c/total for c in counts.values()] if total > 0 else []
            entropy = round(float(-sum(p*math.log(p + 1e-12) for p in probs)) if probs else 0.0, 6)
        else:
            entropy = float(entropy)

    row = [
        now, str(symbol), str(strategy),
        int(group_id) if group_id is not None else 0,
        int(total),
        json.dumps(counts, ensure_ascii=False),
        int(n_unique),
        float(entropy),
        str(note or "")
    ]

    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","total","counts_json","n_unique","entropy","note"])
            w.writerow(row)
        print(f"[📊 라벨분포 로그] {symbol}-{strategy}-g{group_id} → total={total}, classes={n_unique}, H={entropy:.4f}")
    except Exception as e:
        print(f"[⚠️ 라벨분포 로그 실패] {e}")
