# === failure_db.py 최종본 ===
import os
import json
import sqlite3
import datetime
import pytz
from typing import Dict, Any, Iterable, Set, Optional

DB_PATH = "/persistent/logs/failure_patterns.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def now_kst() -> str:
    return datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()

# --------------------------------
# DB 초기화 / 커넥션
# --------------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def ensure_failure_db() -> None:
    """
    실패 패턴 저장용 DB/테이블 보장
    """
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS failure_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            strategy TEXT,
            model_name TEXT,
            predicted_class INTEGER,
            label INTEGER,
            reason TEXT,
            rate REAL,
            return REAL,
            entry_price REAL,
            target_price REAL,
            source TEXT,
            group_id INTEGER,
            hash TEXT UNIQUE,
            feature_vector TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_hash ON failure_patterns(hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_model ON failure_patterns(model_name)")
    conn.commit()
    conn.close()

# --------------------------------
# 중복 확인
# --------------------------------
def check_failure_exists(row: Dict[str, Any]) -> bool:
    """
    동일 (hash, model_name, predicted_class) 가 이미 존재하는지 보수적으로 확인.
    row에는 최소한 hash 또는 (model, predicted_class) 정보가 있어야 함.
    """
    try:
        h = row.get("hash")
        m = row.get("model") or row.get("model_name") or ""
        pc = int(row.get("predicted_class", -1)) if str(row.get("predicted_class", -1)).isdigit() else -1

        conn = get_db_connection()
        cur = conn.cursor()
        if h:  # 해시가 있으면 해시로 1차 확인
            cur.execute("SELECT 1 FROM failure_patterns WHERE hash=? LIMIT 1", (h,))
            if cur.fetchone():
                conn.close()
                return True

        # 해시가 없거나 새로 들어오는 포맷일 수 있으니 보조 조건 확인
        cur.execute(
            "SELECT 1 FROM failure_patterns WHERE model_name=? AND predicted_class=? LIMIT 1",
            (m, pc),
        )
        exists = cur.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False

# --------------------------------
# 실패 기록/조회
# --------------------------------
def insert_failure_record(
    row: Dict[str, Any],
    feature_hash: Optional[str] = None,
    label: Optional[int] = None,
    feature_vector: Optional[Iterable[float]] = None,
) -> None:
    """
    실패 케이스를 DB에 저장(중복 방지).
    - row: symbol/strategy/model/predicted_class/rate/return_value/entry_price/target_price/reason/source/group_id 등 자유
    - feature_hash: 동일 샘플 중복 방지용 키(가능하면 전달 권장)
    - feature_vector: 리스트/ndarray 등 → JSON 문자열로 저장
    """
    ensure_failure_db()

    # 입력 정규화
    symbol   = str(row.get("symbol", "UNKNOWN"))
    strategy = str(row.get("strategy", "알수없음"))
    model    = str(row.get("model") or row.get("model_name") or "")
    pc       = int(row.get("predicted_class", -1)) if str(row.get("predicted_class", -1)).isdigit() else -1
    reason   = str(row.get("reason", "사유없음"))
    rate     = float(row.get("rate", 0.0) if row.get("rate") not in (None, "") else 0.0)
    ret_val  = float(row.get("return_value", row.get("return", 0.0)) or 0.0)
    entry    = float(row.get("entry_price", 0.0) or 0.0)
    target   = float(row.get("target_price", 0.0) or 0.0)
    source   = str(row.get("source", "일반"))
    gid      = int(row.get("group_id", 0)) if str(row.get("group_id", 0)).isdigit() else 0
    lbl      = int(label if label is not None else row.get("label", -1)) if str(label if label is not None else row.get("label", -1)).isdigit() else -1

    h = feature_hash or row.get("hash")
    if not h:
        # 가능한 최소 키로 간이 해시 생성
        base = f"{symbol}|{strategy}|{model}|{pc}|{entry}|{target}"
        import hashlib
        h = hashlib.sha1(base.encode()).hexdigest()

    # 이미 있으면 스킵
    if check_failure_exists({"hash": h, "model": model, "predicted_class": pc}):
        return

    fv_json = None
    if feature_vector is not None:
        try:
            # list 변환 후 JSON 직렬화
            if hasattr(feature_vector, "tolist"):
                feature_vector = feature_vector.tolist()
            fv_json = json.dumps(feature_vector)
        except Exception:
            fv_json = None

    try:
        conn = get_db_connection()
        conn.execute(
            """
            INSERT OR IGNORE INTO failure_patterns
            (timestamp, symbol, strategy, model_name, predicted_class, label,
             reason, rate, return, entry_price, target_price, source, group_id, hash, feature_vector)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now_kst(), symbol, strategy, model, pc, lbl,
                reason, rate, ret_val, entry, target, source, gid, h, fv_json
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # DB 오류는 치명적이지 않게 무시(로그만)
        print(f"[failure_db] insert 실패: {e}")

def load_existing_failure_hashes(limit: int = 50000) -> Set[str]:
    """
    저장되어 있는 실패 해시를 집합으로 반환(중복 방지용)
    """
    try:
        ensure_failure_db()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT hash FROM failure_patterns ORDER BY id DESC LIMIT ?", (int(limit),))
        hashes = {r[0] for r in cur.fetchall() if r and r[0]}
        conn.close()
        return hashes
    except Exception:
        return set()
