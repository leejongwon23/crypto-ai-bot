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

# ✅ 3. 실패 피처 해시 목록 불러오기 (학습 시 중복 판단용) - 수정됨
def load_existing_failure_hashes():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            return set(r[0] for r in rows)  # hash만 추출
    except:
        return set()

# ✅ 4. 실패 사유 자동 분석 함수
def analyze_failure_reason(rate, volatility=None):
    if not isinstance(rate, float):
        return "불명확"
    if abs(rate) < 0.005:
        return "미약한 움직임"
    if rate > 0.02:
        return "과도한 롱 추정 실패"
    if rate < -0.02:
        return "과도한 숏 추정 실패"
    if volatility is not None and volatility > 0.05:
        return "고변동성 구간 실패"
    return "기타 실패"

# ✅ 5. 사유별 실패 클러스터 집계 함수 (선택적 사용)
def group_failures_by_reason(limit=100):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT reason, COUNT(*) as count
                FROM failure_patterns
                GROUP BY reason
                ORDER BY count DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [{"reason": r[0], "count": r[1]} for r in rows]
    except:
        return []
