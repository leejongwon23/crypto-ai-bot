import sqlite3
import os
import json
from collections import defaultdict

def insert_failure_record(row, feature_hash, feature_vector=None, label=None):
    if not isinstance(feature_hash, str) or feature_hash.strip() == "":
        return

    # ✅ feature_vector 저장 전 타입 변환 및 fallback 안전화
    if feature_vector is not None:
        try:
            if hasattr(feature_vector, "tolist"):
                feature_vector = feature_vector.tolist()
            if not isinstance(feature_vector, list):
                feature_vector = None
            else:
                if not all(isinstance(x, list) for x in feature_vector):
                    feature_vector = None
            json.dumps(feature_vector)
        except:
            feature_vector = None

    # ✅ label None 처리 → default -1
    if label is not None:
        try:
            label = int(label)
        except:
            label = -1
    else:
        label = -1

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO failure_patterns (
                    timestamp, symbol, strategy, direction, hash, rate, reason, feature, label
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("timestamp", ""),
                row.get("symbol", ""),
                row.get("strategy", ""),
                row.get("direction", "예측실패"),
                feature_hash,
                float(row.get("rate", 0.0)),
                row.get("reason", ""),
                json.dumps(feature_vector) if feature_vector else None,
                label
            ))
    except Exception as e:
        print(f"[오류] insert_failure_record 실패 → {e}")

def load_existing_failure_hashes():
    import sqlite3
    DB_PATH = "/persistent/logs/failure_patterns.db"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            # ✅ 무결성 검증 추가
            valid_hashes = set(r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != "")
            return valid_hashes
    except Exception as e:
        print(f"[오류] 실패 해시 로드 실패 → {e}")
        return set()
