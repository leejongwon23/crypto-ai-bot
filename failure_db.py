import sqlite3
import os
import json
from collections import defaultdict

DB_PATH = "/persistent/logs/failure_patterns.db"

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
            reason TEXT,
            feature TEXT,
            label INTEGER
        )
        """)

def insert_failure_record(row, feature_hash, feature_vector=None, label=None):
    if not isinstance(feature_hash, str) or feature_hash.strip() == "":
        return
    if feature_vector is not None:
        try:
            json.dumps(feature_vector)
        except:
            feature_vector = None
    if label is not None:
        try:
            label = int(label)
        except:
            label = None
    else:
        label = -1  # ✅ None인 경우 -1로 기본 대입

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
                int(label) if label is not None else -1
            ))
    except Exception as e:
        print(f"[오류] insert_failure_record 실패 → {e}")

def load_existing_failure_hashes():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            # ✅ 무결성 검증 추가
            valid_hashes = set(r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip() != "")
            return valid_hashes
    except Exception as e:
        print(f"[오류] 실패 해시 로드 실패 → {e}")
        return set()

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

def load_failure_hashes_filtered(strategy=None, recent_hours=None):
    try:
        query = "SELECT hash FROM failure_patterns"
        filters, args = [], []

        if strategy:
            filters.append("strategy = ?")
            args.append(strategy)

        if recent_hours:
            filters.append("timestamp >= datetime('now', ?)")
            args.append(f"-{int(recent_hours)} hours")

        if filters:
            query += " WHERE " + " AND ".join(filters)

        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(query, args).fetchall()
            return set(r[0] for r in rows if r and r[0])
    except:
        return set()

def load_failed_feature_data(strategy=None, max_per_class=20):
    result = []
    class_counter = defaultdict(int)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = "SELECT feature, label FROM failure_patterns WHERE feature IS NOT NULL AND label IS NOT NULL"
            if strategy:
                query += " AND strategy = ? ORDER BY id DESC"
                rows = conn.execute(query, (strategy,)).fetchall()
            else:
                query += " ORDER BY id DESC"
                rows = conn.execute(query).fetchall()

            for feat_json, label in rows:
                if not feat_json:
                    continue
                try:
                    label = int(label)
                    if class_counter[label] >= max_per_class:
                        continue
                    feat = json.loads(feat_json)
                    if isinstance(feat, list):
                        # ✅ Tensor 변환을 위한 nested list 보장
                        if all(isinstance(x, list) for x in feat):
                            result.append((feat, label))
                            class_counter[label] += 1
                except:
                    continue
    except Exception as e:
        print(f"[오류] 실패 피처 로딩 실패 → {e}")
    return result
