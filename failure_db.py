import sqlite3
import os
import json
from collections import defaultdict

DB_PATH = "/persistent/logs/failure_patterns.db"

def ensure_failure_db():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    direction TEXT,
                    hash TEXT,
                    model_name TEXT,           -- ✅ 모델명 추가
                    predicted_class INTEGER,   -- ✅ 예측 클래스 추가
                    rate REAL,
                    reason TEXT,
                    feature TEXT,
                    label INTEGER,
                    UNIQUE(hash, model_name, predicted_class) -- ✅ 중복방지 키 수정
                )
            """)
    except Exception as e:
        print(f"[오류] ensure_failure_db 실패 → {e}")

def insert_failure_record(row, feature_hash, feature_vector=None, label=None):
    if not isinstance(feature_hash, str) or feature_hash.strip() == "":
        return

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
            print(f"[경고] feature_vector 변환 실패 → {e}")
            feature_vector = None

    if label is not None:
        try:
            label = int(label)
        except:
            label = -1
    else:
        label = -1

    # ✅ 모델명, predicted_class 추출
    model_name = row.get("model", "unknown")
    predicted_class = row.get("predicted_class", -1)
    try:
        predicted_class = int(predicted_class)
    except:
        predicted_class = -1

    try:
        with sqlite3.connect(DB_PATH) as conn:
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
                json.dumps(feature_vector) if feature_vector is not None else None,
                label
            ))
    except Exception as e:
        print(f"[오류] insert_failure_record 실패 → {e}")

def load_existing_failure_hashes():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            valid_hashes = set(r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != "")
            return valid_hashes
    except Exception as e:
        print(f"[오류] 실패 해시 로드 실패 → {e}")
        return set()
