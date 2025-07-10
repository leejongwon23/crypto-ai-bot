import sqlite3
import os
import json
from collections import defaultdict

DB_PATH = "/persistent/logs/failure_patterns.db"

# ✅ 전역 DB connection singleton
_db_conn = None

def get_db_connection():
    global _db_conn
    if _db_conn is None:
        try:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)  # ✅ multi-thread safe
            print("[✅ DB connection 생성 완료]")
        except Exception as e:
            print(f"[오류] DB connection 생성 실패 → {e}")
            _db_conn = None
    return _db_conn

def ensure_failure_db():
    try:
        conn = get_db_connection()
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
                UNIQUE(hash, model_name, predicted_class)
            )
        """)
        conn.commit()
        print("[✅ ensure_failure_db 완료]")
    except Exception as e:
        print(f"[오류] ensure_failure_db 실패 → {e}")

def insert_failure_record(row, feature_hash, feature_vector=None, label=None):
    if not isinstance(feature_hash, str) or feature_hash.strip() == "":
        print("[❌ insert_failure_record] feature_hash 없음 → 저장 스킵")
        return

    # ✅ feature_vector 처리 강화
    if feature_vector is not None:
        try:
            import numpy as np
            if hasattr(feature_vector, "detach"):
                feature_vector = feature_vector.detach().cpu().numpy()
            if hasattr(feature_vector, "numpy"):
                feature_vector = feature_vector.numpy()
            if hasattr(feature_vector, "tolist"):
                feature_vector = feature_vector.tolist()
            if isinstance(feature_vector, list):
                feature_vector = np.array(feature_vector).flatten().tolist()
            json.dumps(feature_vector)
        except Exception as e:
            print(f"[경고] feature_vector 변환 실패 → zero-vector 대체: {e}")
            feature_vector = [0.0] * 10
    else:
        print("[경고] feature_vector 없음 → zero-vector 대체")
        feature_vector = [0.0] * 10

    # ✅ label 처리 강화
    if label is not None:
        try:
            label = int(label)
        except:
            print("[경고] label 변환 실패 → -1 대체")
            label = -1
    else:
        print("[경고] label None → -1 대체")
        label = -1

    model_name = row.get("model", "unknown")
    predicted_class = row.get("predicted_class", -1)
    try:
        predicted_class = int(predicted_class)
    except:
        predicted_class = -1

    try:
        conn = get_db_connection()
        conn.execute("""
            INSERT OR IGNORE INTO failure_patterns (
                timestamp, symbol, strategy, direction, hash, model_name, predicted_class, rate, reason, feature, label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("timestamp", ""),
            row.get("symbol", ""),
            row.get("strategy", ""),
            row.get("direction", "예측실패"),
            feature_hash,
            model_name,
            predicted_class,
            float(row.get("rate", 0.0)),
            row.get("reason", ""),
            json.dumps(feature_vector),
            label
        ))
        conn.commit()
        print(f"[✅ insert_failure_record] {row.get('symbol')} 저장 완료")
    except Exception as e:
        print(f"[오류] insert_failure_record 실패 → {e}")

def load_existing_failure_hashes():
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
        valid_hashes = set(r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != "")
        return valid_hashes
    except Exception as e:
        print(f"[오류] 실패 해시 로드 실패 → {e}")
        return set()
