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
    import hashlib
    import pandas as pd
    from datetime import datetime

    ensure_failure_db()  # ✅ 테이블 생성 보장

    CSV_PATH = "/persistent/wrong_predictions.csv"

    if not isinstance(row, dict):
        print("[❌ insert_failure_record] row 형식 오류")
        return

    # ✅ feature_hash 생성
    if not feature_hash:
        raw_str = f"{row.get('symbol','')}_{row.get('strategy','')}_{row.get('timestamp','')}_{row.get('predicted_class','')}"
        feature_hash = hashlib.sha256(raw_str.encode()).hexdigest()

    if not isinstance(feature_hash, str) or feature_hash.strip() == "":
        print("[❌ insert_failure_record] feature_hash 없음 → 저장 스킵")
        return

    # ✅ DB 중복 체크
    conn = get_db_connection()
    try:
        exists = conn.execute(
            "SELECT 1 FROM failure_patterns WHERE hash=? AND model_name=? AND predicted_class=?",
            (feature_hash, row.get("model", ""), row.get("predicted_class", -1))
        ).fetchone()
        if exists:
            print(f"[⛔ DB 중복 예측 실패] feature_hash={feature_hash} → 저장 스킵")
            return
    except Exception as e:
        print(f"[⚠️ DB 중복 체크 실패] {e}")

    # ✅ CSV 중복 체크
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
            if "feature_hash" in df.columns and feature_hash in df["feature_hash"].values:
                print(f"[⛔ CSV 중복 예측 실패] feature_hash={feature_hash} → 저장 스킵")
                return
        except Exception as e:
            print(f"[⚠️ CSV 로드 실패] {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    try:
        record_time = datetime.now().isoformat()
        record = {
            "timestamp": record_time,
            "symbol": row.get("symbol"),
            "strategy": row.get("strategy"),
            "direction": row.get("direction", ""),
            "hash": feature_hash,
            "model_name": row.get("model", ""),
            "predicted_class": row.get("predicted_class", -1),
            "rate": row.get("rate", ""),
            "reason": row.get("reason") or "미기록",
            "feature": json.dumps(feature_vector.tolist()) if feature_vector is not None else "",
            "label": row.get("label") if label is None else label
        }

        # ✅ DB 저장
        try:
            conn.execute("""
                INSERT OR IGNORE INTO failure_patterns
                (timestamp, symbol, strategy, direction, hash, model_name, predicted_class, rate, reason, feature, label)
                VALUES (:timestamp, :symbol, :strategy, :direction, :hash, :model_name, :predicted_class, :rate, :reason, :feature, :label)
            """, record)
            conn.commit()
            print(f"[✅ DB 실패 예측 저장] {record['symbol']} {record['strategy']} class={record['predicted_class']}")
        except Exception as e:
            print(f"[⚠️ DB 저장 실패] {e}")

        # ✅ CSV 저장
        csv_record = record.copy()
        csv_record.pop("feature")
        csv_record["feature_hash"] = feature_hash

        if feature_vector is not None:
            for i, v in enumerate(feature_vector.flatten()):
                csv_record[f"f{i}"] = float(v)

        df = pd.concat([df, pd.DataFrame([csv_record])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[✅ CSV 실패 예측 저장] {csv_record['symbol']} {csv_record['strategy']} class={csv_record['predicted_class']}")

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
