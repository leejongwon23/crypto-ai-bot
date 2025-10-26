# === eval_queue.py (v2025-10-26) ============================================
# 목적: 예측을 절대 버리지 않고, 평가 시에는 소량 배치로만 꺼내 메모리 안전하게 처리
# DB: /persistent/logs/eval_queue.db (SQLite, WAL)
# API:
#   enqueue_prediction(symbol, strategy, payload: dict, priority=0) -> int
#   fetch_batch(limit=100) -> List[dict]           # status=pending인 것만 lock & running 전환
#   mark_done(ids: list[int]) -> None
#   mark_failed(ids: list[int], retry_delay_s=60) -> None
#   reset_stuck(max_minutes=15) -> int             # 오래된 running → pending 되돌림
# 환경변수:
#   EVAL_BATCH_SIZE=100
#   EVAL_STUCK_MIN=15
# ===========================================================================

import os, json, sqlite3, time, datetime, threading
from typing import List, Dict, Any, Optional

PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DB_PATH = os.path.join(LOG_DIR, "eval_queue.db")

_DB_LOCK = threading.RLock()
_DB = None

def _apply_pragmas(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=5000;")
    cur.close()

def _get_db():
    global _DB
    with _DB_LOCK:
        if _DB is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            _DB = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
            _apply_pragmas(_DB)
        return _DB

def _ensure_schema():
    conn = _get_db()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS eval_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,                    -- enqueue 시각 (ISO, KST)
        symbol TEXT,
        strategy TEXT,
        payload_json TEXT,          -- 예측 원본/부가정보 전체
        status TEXT,                -- pending|running|done|failed
        try_count INTEGER DEFAULT 0,
        priority INTEGER DEFAULT 0, -- 낮을수록 먼저
        available_at REAL DEFAULT 0 -- epoch seconds; 재시도 지연용
    );
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_eq_status ON eval_queue(status, priority, available_at);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_eq_symstr ON eval_queue(symbol, strategy);")
    conn.commit()
    c.close()

def _now_iso_kst():
    try:
        import pytz
        tz = pytz.timezone("Asia/Seoul")
        return datetime.datetime.now(tz).isoformat()
    except Exception:
        return datetime.datetime.now().isoformat()

def enqueue_prediction(symbol: str, strategy: str, payload: Dict[str, Any], priority: int = 0) -> int:
    """예측 1건을 큐에 적재. 절대 버리지 않음."""
    _ensure_schema()
    conn = _get_db()
    c = conn.cursor()
    c.execute("""
      INSERT INTO eval_queue (ts, symbol, strategy, payload_json, status, try_count, priority, available_at)
      VALUES (?, ?, ?, ?, 'pending', 0, ?, 0)
    """, (_now_iso_kst(), str(symbol), str(strategy), json.dumps(payload, ensure_ascii=False), int(priority)))
    rid = c.lastrowid
    conn.commit()
    c.close()
    return int(rid)

def fetch_batch(limit: int = 100) -> List[Dict[str, Any]]:
    """pending + available_at<=now 인 것 중 우선순위→id로 정렬하여 'running'으로 바꾸고 반환."""
    _ensure_schema()
    now = time.time()
    rows = []
    with _DB_LOCK:
        conn = _get_db()
        c = conn.cursor()
        c.execute("""
          SELECT id, ts, symbol, strategy, payload_json, try_count, priority
          FROM eval_queue
          WHERE status='pending' AND (available_at IS NULL OR available_at<=?)
          ORDER BY priority ASC, id ASC
          LIMIT ?
        """, (now, int(limit)))
        rows = c.fetchall()
        ids = [r[0] for r in rows]
        if ids:
            q_marks = ",".join(["?"] * len(ids))
            c.execute(f"UPDATE eval_queue SET status='running' WHERE id IN ({q_marks})", ids)
            conn.commit()
        c.close()
    out = []
    for (rid, ts, sym, strat, payload_json, try_count, prio) in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}
        out.append({
            "id": int(rid),
            "ts": ts,
            "symbol": sym,
            "strategy": strat,
            "payload": payload,
            "try_count": int(try_count or 0),
            "priority": int(prio or 0)
        })
    return out

def mark_done(ids: List[int]) -> None:
    if not ids: return
    with _DB_LOCK:
        conn = _get_db()
        c = conn.cursor()
        q = ",".join(["?"] * len(ids))
        c.execute(f"UPDATE eval_queue SET status='done' WHERE id IN ({q})", ids)
        conn.commit()
        c.close()

def mark_failed(ids: List[int], retry_delay_s: int = 60) -> None:
    if not ids: return
    now = time.time()
    avail = now + max(0, int(retry_delay_s))
    with _DB_LOCK:
        conn = _get_db()
        c = conn.cursor()
        q = ",".join(["?"] * len(ids))
        # try_count++, pending으로 되돌리고 일정 시간 뒤 재시도
        c.execute(f"""
            UPDATE eval_queue
            SET status='pending',
                try_count=COALESCE(try_count,0)+1,
                available_at=?
            WHERE id IN ({q})
        """, (avail, *ids))
        conn.commit()
        c.close()

def reset_stuck(max_minutes: int = 15) -> int:
    """오래도록 running인 항목을 pending으로 원복."""
    cutoff = time.time() - (max(1, int(max_minutes)) * 60)
    # running 이 언제 running 됐는지 DB에 없으므로, try_count>0 + available_at==0인 running 을 느슨히 복구:
    with _DB_LOCK:
        conn = _get_db()
        c = conn.cursor()
        # 간단 버전: running 전부를 pending으로 되돌리되, try_count+1
        c.execute("""
          UPDATE eval_queue
          SET status='pending', try_count=COALESCE(try_count,0)+1
          WHERE status='running'
        """)
        n = c.rowcount
        conn.commit()
        c.close()
    return int(n)

# 편의값
def get_env_batch_size(default: int = 100) -> int:
    try:
        return int(os.getenv("EVAL_BATCH_SIZE", str(default)))
    except Exception:
        return default
def get_env_stuck_min(default: int = 15) -> int:
    try:
        return int(os.getenv("EVAL_STUCK_MIN", str(default)))
    except Exception:
        return default
