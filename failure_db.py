# === failure_db.py (patched, req-unit connections + invalid gate) ===
import sqlite3
import os
import json
import hashlib
from threading import Lock
from datetime import datetime

DB_PATH = "/persistent/logs/failure_patterns.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 커넥션 팩토리 (요청 단위로 열고 닫음)
#  - autocommit 모드(isolation_level=None) + 필요한 곳에서 BEGIN/COMMIT
#  - WAL / NORMAL / busy_timeout 설정
# ──────────────────────────────────────────────────────────────
def open_conn():
    conn = sqlite3.connect(DB_PATH, timeout=5.0, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

# ──────────────────────────────────────────────────────────────
# 스키마 보장 (프로세스 생애 동안 1회만)
# ──────────────────────────────────────────────────────────────
_schema_ready = False
_schema_lock = Lock()

def ensure_failure_db():
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        try:
            with open_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS failure_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        symbol TEXT,
                        strategy TEXT,
                        direction TEXT,
                        hash TEXT,
                        model_name TEXT,
                        predicted_class INTEGER,
                        rate REAL,
                        reason TEXT,
                        feature TEXT,
                        label INTEGER,
                        context TEXT,
                        UNIQUE(hash, model_name, predicted_class)
                    )
                """)
                # autocommit 모드이므로 명시 커밋 필요 없음
            _schema_ready = True
            print("[failure_db] ✅ ensure_failure_db OK")
        except Exception as e:
            print(f"[failure_db] ❌ ensure_failure_db error: {e}")

# ──────────────────────────────────────────────────────────────
# 존재 확인 (중복 방지)
#  - 요청 단위 커넥션 사용
# ──────────────────────────────────────────────────────────────
def _build_hash_from_row(row, feature_hash=None, label=None):
    if feature_hash:
        return feature_hash
    sym = row.get("symbol", "")
    strat = row.get("strategy", "")
    mdl = row.get("model", "")
    pcls = row.get("predicted_class", -1)
    lab = label if label is not None else row.get("label", -1)
    rt = row.get("rate", "")
    raw = f"{sym}_{strat}_{mdl}_{pcls}_{lab}_{rt}"
    return hashlib.sha256(raw.encode()).hexdigest()

def check_failure_exists(row_or_hash, model_name=None, predicted_class=None):
    """
    row_or_hash 가 dict면 내부 키로 해시 계산/조회,
    str 이면 그대로 hash 로 간주.
    """
    ensure_failure_db()
    try:
        if isinstance(row_or_hash, dict):
            h = row_or_hash.get("hash") or row_or_hash.get("feature_hash")
            h = h or _build_hash_from_row(row_or_hash)
            mdl = row_or_hash.get("model", "") or (model_name or "")
            pcls = row_or_hash.get("predicted_class", -1) if predicted_class is None else int(predicted_class)
        else:
            h = str(row_or_hash)
            mdl = model_name or ""
            pcls = -1 if predicted_class is None else int(predicted_class)

        with open_conn() as conn:
            cur = conn.execute(
                "SELECT 1 FROM failure_patterns WHERE hash=? AND model_name=? AND predicted_class=? LIMIT 1",
                (h, mdl, pcls),
            )
            return cur.fetchone() is not None
    except Exception as e:
        print(f"[failure_db] check_failure_exists error: {e}")
        return False

# ──────────────────────────────────────────────────────────────
# 저장
#  - 체크+인서트를 같은 트랜잭션으로(경합 제거)
#  - 요청 단위 커넥션 + BEGIN IMMEDIATE
# ──────────────────────────────────────────────────────────────
_write_lock = Lock()  # 파이썬 레벨 락으로 간단한 경쟁 방지

# 🚫 추가: invalid 케이스 차단 게이트
_INVALID_REASON_KEYS = (
    "invalid",              # e.g. invalid_entry_or_label
    "timestamp_parse_error",
    "no_price_data",
    "no_data_until_deadline",
    "exception:"            # parsing/기타 예외
)

def _should_block(row: dict, label_val) -> bool:
    try:
        # label 미기록 또는 음수
        if label_val is None or int(label_val) < 0:
            return True
    except Exception:
        return True

    # entry_price<=0 이면 차단 (prediction_log 에서 invalid 로 본 건 저장 X)
    try:
        ep = float(row.get("entry_price", 0) or 0)
        if ep <= 0:
            return True
    except Exception:
        return True

    # symbol/strategy 필수
    if not str(row.get("symbol", "")).strip() or not str(row.get("strategy", "")).strip():
        return True

    # 명시적 invalid/status
    status = str(row.get("status", "")).strip().lower()
    if status == "invalid":
        return True

    # 사유(reason)에 invalid/exception 류 키워드 포함 시 차단
    reason = str(row.get("reason", "")).strip().lower()
    for key in _INVALID_REASON_KEYS:
        if key in reason:
            return True

    return False

def insert_failure_record(row, feature_hash=None, feature_vector=None, label=None, context="evaluation"):
    """
    실패 예측을 기록한다.
    context: "evaluation" | "prediction" 등
    """
    ensure_failure_db()

    if not isinstance(row, dict):
        print("[failure_db] ❌ row must be dict")
        return

    # label 정규화(음수/미기록 허용 → 차단 게이트에서 처리)
    try:
        label_val = label if label is not None else row.get("label", -1)
        label_int = int(label_val)
    except Exception:
        label_int = -1

    # 🚫 invalid 차단
    if _should_block(row, label_int):
        print("[failure_db] ⛔ blocked invalid failure record (not saved)")
        return

    # hash
    feature_hash = _build_hash_from_row(row, feature_hash=feature_hash, label=label_int)

    mdl_name = row.get("model", "")
    pcls = int(row.get("predicted_class", -1))

    # feature vector serialize
    try:
        import numpy as np
        def to_list_safe(x):
            if x is None: return []
            if isinstance(x, np.ndarray): return x.flatten().astype(float).tolist()
            if isinstance(x, (list, tuple)):
                out = []
                for v in x:
                    if isinstance(v, (int, float, np.integer, np.floating)): out.append(float(v))
                    else: out.append(None)
                return out
            if isinstance(x, (int, float, np.integer, np.floating)): return [float(x)]
            try: return list(x)
            except Exception: return []
        feature_json = json.dumps(to_list_safe(feature_vector), ensure_ascii=False)
    except Exception:
        feature_json = "[]"

    rec = {
        "timestamp": row.get("timestamp") or datetime.utcnow().isoformat(),
        "symbol": row.get("symbol", ""),
        "strategy": row.get("strategy", ""),
        "direction": row.get("direction", ""),
        "hash": feature_hash,
        "model_name": mdl_name,
        "predicted_class": pcls,
        "rate": row.get("rate", 0.0) if row.get("rate") not in (None, "") else 0.0,
        "reason": row.get("reason", "미기록"),
        "feature": feature_json,
        "label": label_int,
        "context": context,
    }

    # 동일 트랜잭션으로 중복 체크 + 삽입
    with _write_lock:
        try:
            with open_conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "SELECT 1 FROM failure_patterns WHERE hash=? AND model_name=? AND predicted_class=? LIMIT 1",
                    (rec["hash"], rec["model_name"], rec["predicted_class"]),
                )
                if cur.fetchone():
                    conn.execute("COMMIT")
                    print(f"[failure_db] ⏭️ skip duplicate hash={rec['hash']}")
                    return

                conn.execute(
                    """
                    INSERT INTO failure_patterns
                    (timestamp, symbol, strategy, direction, hash, model_name, predicted_class,
                     rate, reason, feature, label, context)
                    VALUES (:timestamp, :symbol, :strategy, :direction, :hash, :model_name, :predicted_class,
                            :rate, :reason, :feature, :label, :context)
                    """,
                    rec,
                )
                conn.execute("COMMIT")
                print(f"[failure_db] ✅ saved {rec['symbol']} {rec['strategy']} cls={rec['predicted_class']} ctx={context}")
        except Exception as e:
            # 트랜잭션 에러 시 롤백 시도
            try:
                with open_conn() as conn:
                    conn.execute("ROLLBACK")
            except Exception:
                pass
            print(f"[failure_db] ❌ insert error: {e}")

# ──────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────
def load_failure_samples(limit=1000):
    """최근 실패 샘플 일부 반환 (메타학습/분석용)"""
    ensure_failure_db()
    try:
        with open_conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, symbol, strategy, model_name, predicted_class, rate, reason, feature, label
                FROM failure_patterns
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        out = []
        for (ts, sym, strat, mdl, pcls, rate, reason, feature, label) in rows:
            try:
                fv = json.loads(feature) if feature else []
            except Exception:
                fv = []
            out.append({
                "timestamp": ts,
                "symbol": sym,
                "strategy": strat,
                "model": mdl,
                "predicted_class": pcls,
                "rate": rate,
                "reason": reason,
                "feature": fv,
                "label": label,
            })
        return out
    except Exception as e:
        print(f"[failure_db] ❌ load_failure_samples error: {e}")
        return []

def load_existing_failure_hashes():
    ensure_failure_db()
    try:
        with open_conn() as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
        return {r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != ""}
    except Exception as e:
        print(f"[failure_db] ❌ load_existing_failure_hashes error: {e}")
        return set()
