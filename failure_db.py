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

    ensure_failure_db()  # ✅ 테이블 생성 보장
    CSV_PATH = "/persistent/wrong_predictions.csv"

    if not isinstance(row, dict):
        print("[❌ insert_failure_record] row 형식 오류 → 저장 스킵")
        return

    # ✅ feature_hash 생성 (고유성 강화)
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

    # ✅ feature_vector 타입 안전화
    if feature_vector is None:
        feature_vector = []
    elif isinstance(feature_vector, np.ndarray):
        feature_vector = feature_vector.flatten().tolist()
    elif not isinstance(feature_vector, list):
        try:
            feature_vector = list(feature_vector)
        except:
            feature_vector = []

    # ✅ DB/CSV 중복 체크
    try:
        # DB 중복 체크
        conn = get_db_connection()
        exists_db = conn.execute(
            "SELECT 1 FROM failure_patterns WHERE hash=? AND model_name=? AND predicted_class=?",
            (feature_hash, row.get("model", ""), row.get("predicted_class", -1))
        ).fetchone()

        # CSV 중복 체크
        exists_csv = False
        if os.path.exists(CSV_PATH):
            try:
                df_csv = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
                if "feature_hash" in df_csv.columns and feature_hash in df_csv["feature_hash"].values:
                    exists_csv = True
            except Exception as e:
                print(f"[⚠️ CSV 로드 실패] {e}")

        # ✅ 둘 중 하나라도 중복이면 스킵
        if exists_db or exists_csv:
            print(f"[⛔ SKIP-중복] feature_hash={feature_hash}")
            return

    except Exception as e:
        print(f"[⚠️ 중복 체크 실패] {e}")

    # ✅ 기록 데이터 준비
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
        csv_record.pop("feature")  # feature는 JSON 대신 개별 컬럼
        csv_record["feature_hash"] = feature_hash

        for i, v in enumerate(feature_vector):
            try:
                csv_record[f"f{i}"] = float(v)
            except:
                csv_record[f"f{i}"] = v

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
