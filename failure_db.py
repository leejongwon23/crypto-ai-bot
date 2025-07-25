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

def insert_failure_record(row, feature_hash=None, feature_vector=None, label=None):
    import os, hashlib
    import pandas as pd
    from datetime import datetime

    CSV_PATH = "/persistent/wrong_predictions.csv"

    if not isinstance(row, dict):
        print("[❌ insert_failure_record] row 형식 오류")
        return

    if not feature_hash:
        raw_str = f"{row.get('symbol','')}_{row.get('strategy','')}_{row.get('timestamp','')}_{row.get('predicted_class','')}"
        feature_hash = hashlib.sha256(raw_str.encode()).hexdigest()

    if not isinstance(feature_hash, str) or feature_hash.strip() == "":
        print("[❌ insert_failure_record] feature_hash 없음 → 저장 스킵")
        return

    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
            if "feature_hash" in df.columns and feature_hash in df["feature_hash"].values:
                print(f"[⛔ 중복 예측 실패] feature_hash 중복 → 저장 스킵")
                return
        except Exception as e:
            print(f"[⚠️ CSV 로드 실패] {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    try:
        is_evo = row.get("source") == "evo_meta"

        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": row.get("symbol"),
            "strategy": row.get("strategy"),
            "model": row.get("model", ""),
            "predicted_class": row.get("predicted_class", -1),
            "label": row.get("label") if label is None else label,
            "feature_hash": feature_hash,
            "rate": row.get("rate", ""),
            "return_value": row.get("return", ""),
            "reason": row.get("reason") or "미기록",
            "evo_meta_strategy": row.get("strategy") if is_evo else ""
        }

        if feature_vector is not None:
            for i, v in enumerate(feature_vector.flatten()):
                record[f"f{i}"] = float(v)

        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[✅ 실패 예측 저장 완료] {record['symbol']} {record['strategy']} → class={record['predicted_class']} ≠ label={record['label']}")

    except Exception as e:
        print(f"[❌ insert_failure_record 예외] {type(e).__name__}: {e}")

def load_existing_failure_hashes():
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
        valid_hashes = set(r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != "")
        return valid_hashes
    except Exception as e:
        print(f"[오류] 실패 해시 로드 실패 → {e}")
        return set()
