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
    import hashlib, os, json
    import pandas as pd
    import numpy as np
    from datetime import datetime

    ensure_failure_db()
    CSV_PATH = "/persistent/wrong_predictions.csv"

    # ✅ 기본 검증
    if not isinstance(row, dict):
        print("[❌ insert_failure_record] row 형식 오류 → 저장 스킵")
        return
    if row.get("success") is True:
        print("[⛔ SKIP] 성공 예측 → 실패 기록 안 함")
        return

    # ✅ feature_hash 생성
    if not feature_hash:
        raw_str = (
            f"{row.get('symbol','')}_{row.get('strategy','')}_"
            f"{row.get('model','')}_{row.get('predicted_class','')}_"
            f"{row.get('label','')}_{row.get('rate','')}"
        )
        feature_hash = hashlib.sha256(raw_str.encode()).hexdigest()

    if not isinstance(feature_hash, str) or not feature_hash.strip():
        print("[❌ insert_failure_record] feature_hash 없음 → 저장 스킵")
        return

    # ✅ 타입 안전 변환
    def to_list_safe(x):
        if x is None:
            return []
        if isinstance(x, np.ndarray):
            return x.flatten().astype(float).tolist()
        if isinstance(x, (list, tuple)):
            return [float(v) if isinstance(v, (int, float, np.integer, np.floating)) else np.nan for v in x]
        if isinstance(x, (int, float, np.integer, np.floating)):
            return [float(x)]
        try:
            return list(x)
        except:
            return []

    feature_vector = to_list_safe(feature_vector)

    # ✅ DB & CSV 중복 체크
    try:
        conn = get_db_connection()
        exists_db = conn.execute(
            "SELECT 1 FROM failure_patterns WHERE hash=? AND model_name=? AND predicted_class=?",
            (feature_hash, row.get("model", ""), row.get("predicted_class", -1))
        ).fetchone()

        exists_csv = False
        if os.path.exists(CSV_PATH):
            try:
                df_csv = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
                if all(col in df_csv.columns for col in ["feature_hash", "model_name", "predicted_class"]):
                    exists_csv = not df_csv[
                        (df_csv["feature_hash"] == feature_hash) &
                        (df_csv["model_name"] == row.get("model", "")) &
                        (df_csv["predicted_class"] == row.get("predicted_class", -1))
                    ].empty
            except Exception as e:
                print(f"[⚠️ CSV 로드 실패] {e}")

        if exists_db or exists_csv:
            print(f"[⛔ SKIP-중복] feature_hash={feature_hash}")
            return

    except Exception as e:
        print(f"[⚠️ 중복 체크 실패] {e}")

    # ✅ 기록 데이터
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
        "feature": json.dumps(feature_vector, ensure_ascii=False),
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
        print(f"[✅ SAVE-DB] {record['symbol']} {record['strategy']} class={record['predicted_class']}")
    except Exception as e:
        print(f"[⚠️ DB 저장 실패] {e}")

    # ✅ CSV 저장
    try:
        csv_record = record.copy()
        csv_record.pop("feature")
        csv_record["feature_hash"] = feature_hash

        for i, v in enumerate(feature_vector):
            csv_record[f"f{i}"] = np.nan if v is None else float(v)

        if os.path.exists(CSV_PATH):
            df_csv = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        else:
            df_csv = pd.DataFrame()

        df_csv = pd.concat([df_csv, pd.DataFrame([csv_record])], ignore_index=True)
        df_csv.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[✅ SAVE-CSV] {csv_record['symbol']} {csv_record['strategy']} class={csv_record['predicted_class']}")
    except Exception as e:
        print(f"[❌ CSV 저장 실패] {type(e).__name__}: {e}")

def load_existing_failure_hashes():
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
        valid_hashes = set(r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != "")
        return valid_hashes
    except Exception as e:
        print(f"[오류] 실패 해시 로드 실패 → {e}")
        return set()
