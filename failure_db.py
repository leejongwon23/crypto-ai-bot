# === failure_db.py (최종본) ===
import sqlite3
import os
import json
from collections import defaultdict
from datetime import datetime

DB_PATH = "/persistent/logs/failure_patterns.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 커넥션 (싱글톤)
# ──────────────────────────────────────────────────────────────
_conn = None

def get_db_connection():
    global _conn
    if _conn is None:
        try:
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.execute("PRAGMA journal_mode=WAL;")
            _conn.execute("PRAGMA synchronous=NORMAL;")
            print("[failure_db] ✅ DB connection ready")
        except Exception as e:
            print(f"[failure_db] ❌ DB connect error: {e}")
            _conn = None
    return _conn

# ──────────────────────────────────────────────────────────────
# 스키마 보장
# ──────────────────────────────────────────────────────────────
def ensure_failure_db():
    try:
        conn = get_db_connection()
        if conn is None:
            print("[failure_db] ❌ no connection")
            return
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
        conn.commit()
        print("[failure_db] ✅ ensure_failure_db OK")
    except Exception as e:
        print(f"[failure_db] ❌ ensure_failure_db error: {e}")

# ──────────────────────────────────────────────────────────────
# 존재 확인 (중복 방지)
# ──────────────────────────────────────────────────────────────
def check_failure_exists(row_or_hash, model_name=None, predicted_class=None):
    """
    row_or_hash 가 dict면 내부 키로 해시 계산/조회,
    str 이면 그대로 hash 로 간주.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False

        if isinstance(row_or_hash, dict):
            h = row_or_hash.get("hash") or row_or_hash.get("feature_hash")
            if not h:
                # 거칠게 재구성
                sym = row_or_hash.get("symbol", "")
                strat = row_or_hash.get("strategy", "")
                mdl = row_or_hash.get("model", "") or model_name or ""
                pcls = row_or_hash.get("predicted_class", -1) if predicted_class is None else predicted_class
                lab = row_or_hash.get("label", -1)
                rt = row_or_hash.get("rate", "")
                raw = f"{sym}_{strat}_{mdl}_{pcls}_{lab}_{rt}"
                import hashlib
                h = hashlib.sha256(raw.encode()).hexdigest()
            mdl = row_or_hash.get("model", "") or model_name or ""
            pcls = row_or_hash.get("predicted_class", -1) if predicted_class is None else predicted_class
        else:
            h = str(row_or_hash)
            mdl = model_name or ""
            pcls = -1 if predicted_class is None else int(predicted_class)

        cur = conn.execute(
            "SELECT 1 FROM failure_patterns WHERE hash=? AND model_name=? AND predicted_class=? LIMIT 1",
            (h, mdl, pcls)
        )
        return cur.fetchone() is not None
    except Exception as e:
        print(f"[failure_db] check_failure_exists error: {e}")
        return False

# ──────────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────────
def insert_failure_record(row, feature_hash=None, feature_vector=None, label=None, context="evaluation"):
    """
    실패 예측을 기록한다.
    context: "evaluation" | "prediction" 등
    """
    ensure_failure_db()

    # 방어적 체크
    if not isinstance(row, dict):
        print("[failure_db] ❌ row must be dict")
        return

    # hash 생성
    if not feature_hash:
        sym = row.get("symbol", "")
        strat = row.get("strategy", "")
        mdl = row.get("model", "")
        pcls = row.get("predicted_class", -1)
        lab = label if label is not None else row.get("label", -1)
        rt = row.get("rate", "")
        raw = f"{sym}_{strat}_{mdl}_{pcls}_{lab}_{rt}"
        import hashlib
        feature_hash = hashlib.sha256(raw.encode()).hexdigest()

    mdl_name = row.get("model", "")
    pcls = int(row.get("predicted_class", -1))
    if check_failure_exists(feature_hash, model_name=mdl_name, predicted_class=pcls):
        print(f"[failure_db] ⏭️ skip duplicate hash={feature_hash}")
        return

    # feature vector serialize
    try:
        import numpy as np
        def to_list_safe(x):
            if x is None:
                return []
            if isinstance(x, np.ndarray):
                return x.flatten().astype(float).tolist()
            if isinstance(x, (list, tuple)):
                return [float(v) if isinstance(v, (int, float, np.integer, np.floating)) else None for v in x]
            if isinstance(x, (int, float, np.integer, np.floating)):
                return [float(x)]
            try:
                return list(x)
            except Exception:
                return []
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
        "rate": row.get("rate", 0.0) if row.get("rate") not in [None, ""] else 0.0,
        "reason": row.get("reason", "미기록"),
        "feature": feature_json,
        "label": int(label if label is not None else row.get("label", -1)) if str(label if label is not None else row.get("label", -1)).isdigit() else -1,
        "context": context
    }

    try:
        conn = get_db_connection()
        conn.execute("""
            INSERT OR IGNORE INTO failure_patterns
            (timestamp, symbol, strategy, direction, hash, model_name, predicted_class, rate, reason, feature, label, context)
            VALUES (:timestamp, :symbol, :strategy, :direction, :hash, :model_name, :predicted_class, :rate, :reason, :feature, :label, :context)
        """, rec)
        conn.commit()
        print(f"[failure_db] ✅ saved {rec['symbol']} {rec['strategy']} cls={rec['predicted_class']} ctx={context}")
    except Exception as e:
        print(f"[failure_db] ❌ insert error: {e}")

# ──────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────
def load_failure_samples(limit=1000):
    """
    최근 실패 샘플 일부 반환 (메타학습/분석용)
    """
    try:
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT timestamp, symbol, strategy, model_name, predicted_class, rate, reason, feature, label
            FROM failure_patterns
            ORDER BY id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()

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
                "label": label
            })
        return out
    except Exception as e:
        print(f"[failure_db] ❌ load_failure_samples error: {e}")
        return []

def load_existing_failure_hashes():
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
        return set(r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != "")
    except Exception as e:
        print(f"[failure_db] ❌ load_existing_failure_hashes error: {e}")
        return set()
