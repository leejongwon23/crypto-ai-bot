# 📁 파일명: failure_db.py (YOPO의 실패기록 전용 DB)

import sqlite3
import os

# ✅ DB 파일 경로
DB_PATH = "/persistent/logs/failure_patterns.db"

# ✅ 1. DB 초기화 함수 (최초 실행 시 테이블 생성)
def ensure_failure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS failure_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            strategy TEXT,
            direction TEXT,
            hash TEXT UNIQUE,
            rate REAL,
            reason TEXT
        )
        """)

# ✅ 2. 실패 기록 저장 함수 (중복되면 자동 무시)
def insert_failure_record(row, feature_hash):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO failure_patterns (timestamp, symbol, strategy, direction, hash, rate, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row["timestamp"],
            row["symbol"],
            row["strategy"],
            row.get("direction", "예측실패"),
            feature_hash,
            float(row.get("rate", 0.0)),
            row.get("reason", "")
        ))

# ✅ 3. 실패 피처 해시 목록 불러오기 (학습 시 중복 판단용)
def load_existing_failure_hashes():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT symbol, strategy, direction, hash FROM failure_patterns").fetchall()
            return set((r[0], r[1], r[2], r[3]) for r in rows)
    except:
        return set()
