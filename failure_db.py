# === failure_db.py (v2025-09-30b, 견고화 반영본) ===
# 실패 레코드 표준화 + 안전 CSV/SQLite 동시 기록
# - wrong_predictions.csv 스키마를 로더(wrong_data_loader.py)가 기대하는 최소 컬럼을 보장
# - 중복가드: feature_hash / 근접타임스탬프+키 기준
# - 원인태깅: negative_label, nan_label, prob_nan, class_out_of_range, bounds_mismatch 등
# - 즉시 경보: 콘솔(필수) + /persistent/logs/alerts.log 파일(선택)
# - 2025-09-30: SQL 조건/CSV-락/SQLite 재시도 패치

import os, csv, json, math, hashlib, time, threading, datetime, pytz
import sqlite3
from typing import Any, Dict, Optional, Iterable

import pandas as pd
import numpy as np

DIR = "/persistent"
LOG_DIR = os.path.join(DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# CSV 파일(로더가 읽는 경로)
WRONG_CSV = os.path.join(DIR, "wrong_predictions.csv")

# SQLite (요약/쿼리용)
DB_PATH = os.path.join(LOG_DIR, "failure_records.db")

# 알림 로그
ALERT_LOG = os.path.join(LOG_DIR, "alerts.log")

# CSV 헤더(최소 보장 컬럼 + 확장)
WRONG_HEADERS = [
    "timestamp","symbol","strategy","predicted_class","label",
    "model","group_id","entry_price","target_price","return_value",
    "reason","context","note","regime","meta_choice",
    "raw_prob","calib_prob","calib_ver",
    "feature_hash","feature_vector","source","source_exchange"
]

# ------------- 공용 유틸 -------------
def _now_kst_iso():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()

def _sha1_of_list(v: Iterable[float]) -> str:
    try:
        xs = [round(float(x), 4) for x in v]
    except Exception:
        xs = []
    joined = ",".join(map(str, xs))
    return hashlib.sha1(joined.encode()).hexdigest()

def _safe_float(x, default=""):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _safe_int(x, default=""):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(float(x))
    except Exception:
        return default

def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (dict, list, tuple)):
            try:
                out[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[k] = str(v)
        else:
            out[k] = v
    return out

# ------------- 파일 보장 -------------
def _ensure_wrong_csv():
    os.makedirs(os.path.dirname(WRONG_CSV), exist_ok=True)
    if not os.path.exists(WRONG_CSV) or os.path.getsize(WRONG_CSV) == 0:
        with open(WRONG_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(WRONG_HEADERS)

def _apply_sqlite_pragmas(conn):
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.close()
    except Exception:
        pass

_db_lock = threading.RLock()
_db = None

def _connect_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    _apply_sqlite_pragmas(conn)
    return conn

def _get_db():
    global _db
    with _db_lock:
        if _db is None:
            _db = _connect_db()
        return _db

def ensure_failure_db():
    """CSV 헤더와 SQLite 테이블을 보장"""
    _ensure_wrong_csv()
    try:
        with _db_lock:
            conn = _get_db()
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    predicted_class INTEGER,
                    label INTEGER,
                    model TEXT,
                    group_id INTEGER,
                    reason TEXT,
                    context TEXT,
                    regime TEXT,
                    raw_prob REAL,
                    calib_prob REAL,
                    feature_hash TEXT,
                    UNIQUE(ts, symbol, strategy, predicted_class, feature_hash)
                );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_failures_ss ON failures(symbol,strategy);")
            conn.commit()
            c.close()
    except Exception as e:
        print(f"[failure_db] ensure_failure_db 예외: {e}")

# ------------- 중복 체크 -------------
def _candidate_hash(record: Dict[str, Any]) -> str:
    if record is None:
        return "none"
    fh = str(record.get("feature_hash") or "").strip()
    if fh:
        return fh
    fv = record.get("feature_vector")
    if isinstance(fv, (list, tuple, np.ndarray)):
        try:
            arr = np.array(fv, dtype=float).reshape(-1)
        except Exception:
            arr = []
        return _sha1_of_list(arr)
    return "none"

def check_failure_exists(row: Dict[str, Any]) -> bool:
    """최근(±90분) 동일 키의 실패 레코드 존재 여부"""
    try:
        ensure_failure_db()

        ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
        if pd.isna(ts):
            return False
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Seoul")
        else:
            ts = ts.tz_convert("Asia/Seoul")

        ts_min_ts = ts - pd.Timedelta(minutes=90)
        ts_max_ts = ts + pd.Timedelta(minutes=90)
        ts_min = ts_min_ts.isoformat()
        ts_max = ts_max_ts.isoformat()

        sym = str(row.get("symbol", ""))
        strat = str(row.get("strategy", ""))
        pcls = _safe_int(row.get("predicted_class"), default="")
        fh = _candidate_hash(row)

        # 1) SQLite 조회
        with _db_lock:
            conn = _get_db()
            c = conn.cursor()
            c.execute("""
                SELECT 1 FROM failures
                 WHERE symbol=? AND strategy=?
                   AND ts BETWEEN ? AND ?
                   AND (? = '' OR predicted_class = ?)
                   AND (? = 'none' OR feature_hash = ?)
                 LIMIT 1;
            """, (sym, strat, ts_min, ts_max,
                  "" if pcls == "" else None, pcls if pcls != "" else None,
                  fh, fh))
            hit = c.fetchone()
            c.close()
        if hit:
            return True

        # 2) CSV 최근 부분 스캔
        if os.path.exists(WRONG_CSV):
            try:
                tail_rows = 20000
                use = ["timestamp", "symbol", "strategy", "predicted_class", "feature_hash"]
                df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig", usecols=lambda c: c in use)
                if len(df) > tail_rows:
                    df = df.tail(tail_rows)

                df = df[(df["symbol"] == sym) & (df["strategy"] == strat)].copy()
                if df.empty:
                    return False

                t = pd.to_datetime(df["timestamp"], errors="coerce")
                t = t.dt.tz_localize("Asia/Seoul", nonexistent="NaT", ambiguous="NaT", errors="ignore")
                m = (t >= ts_min_ts) & (t <= ts_max_ts)
                df = df[m]

                if df.empty:
                    return False

                if fh != "none" and "feature_hash" in df.columns:
                    if (df["feature_hash"].astype(str) == fh).any():
                        return True

                if pcls != "" and "predicted_class" in df.columns:
                    pc = pd.to_numeric(df["predicted_class"], errors="coerce")
                    if (pc == int(pcls)).any():
                        return True

                return False
            except Exception:
                return False

        return False
    except Exception:
        return False

# ------------- 기존 실패 해시 로딩 -------------
def load_existing_failure_hashes() -> set:
    ensure_failure_db()
    hashes = set()
    try:
        if os.path.exists(WRONG_CSV):
            for chunk in pd.read_csv(WRONG_CSV, encoding="utf-8-sig", usecols=["feature_hash"], chunksize=20000):
                if "feature_hash" in chunk.columns:
                    hashes.update([str(h) for h in chunk["feature_hash"].dropna().astype(str) if str(h)])
    except Exception:
        pass
    return hashes

# ------------- 원인 태깅 -------------
def _classify_failure_reason(rec: Dict[str, Any]) -> str:
    try:
        lbl = rec.get("label", None)
        if lbl is not None:
            try:
                li = int(lbl)
                if li < 0:
                    return "negative_label"
            except Exception:
                return "nan_label"
        rp = rec.get("raw_prob", None)
        cp = rec.get("calib_prob", None)
        if rp not in (None, "") and (math.isnan(float(rp)) or math.isinf(float(rp))):
            return "prob_nan"
        if cp not in (None, "") and (math.isnan(float(cp)) or math.isinf(float(cp))):
            return "prob_nan"
        rs = str(rec.get("reason","")).strip().lower()
        if "bounds" in rs or "range" in rs:
            return "bounds_mismatch"
        if "class_out_of_range" in rs:
            return "class_out_of_range"
        return rs if rs else "unknown"
    except Exception:
        return "unknown"

# ------------- 경보 -------------
def _emit_alert(msg: str):
    try:
        print(f"🔴 [ALERT] {msg}")
        with open(ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{_now_kst_iso()} {msg}\n")
    except Exception:
        pass

# ------------- CSV append with lock/retry -------------
def _append_wrong_csv_row(row: Dict[str, Any], max_retries: int = 5, sleep_sec: float = 0.05):
    _ensure_wrong_csv()
    attempt = 0
    while True:
        try:
            try:
                import fcntl
                with open(WRONG_CSV, "a", newline="", encoding="utf-8-sig") as f:
                    try: fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    except Exception: pass
                    w = csv.DictWriter(f, fieldnames=WRONG_HEADERS)
                    w.writerow({k: row.get(k, "") for k in WRONG_HEADERS})
                    try: f.flush(); os.fsync(f.fileno())
                    except Exception: pass
            except Exception:
                with open(WRONG_CSV, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=WRONG_HEADERS)
                    w.writerow({k: row.get(k, "") for k in WRONG_HEADERS})
            return
        except Exception:
            attempt += 1
            if attempt >= max_retries:
                raise
            time.sleep(sleep_sec)

# ------------- 메인 API -------------
def insert_failure_record(record: Dict[str, Any],
                          feature_hash: Optional[str] = None,
                          label: Optional[int] = None,
                          feature_vector: Optional[Iterable[float]] = None,
                          context: Optional[str] = None) -> bool:
    try:
        ensure_failure_db()
        rec = dict(record or {})
        rec = _sanitize_dict(rec)

        ts = rec.get("timestamp") or _now_kst_iso()
        sym = str(rec.get("symbol","UNKNOWN"))
        strat = str(rec.get("strategy","알수없음"))
        pcls = _safe_int(rec.get("predicted_class"), default=-1)
        lbl  = label if label is not None else _safe_int(rec.get("label"), default="")

        fv = feature_vector if feature_vector is not None else rec.get("feature_vector", None)
        if isinstance(fv, str):
            try: fv = json.loads(fv)
            except Exception: fv = []
        fh = feature_hash or rec.get("feature_hash") or (_sha1_of_list(fv) if isinstance(fv,(list,tuple,np.ndarray)) else "none")

        row = {
            "timestamp": ts, "symbol": sym, "strategy": strat,
            "predicted_class": pcls if pcls != "" else -1,
            "label": lbl if lbl != "" else -1,
            "model": rec.get("model",""),
            "group_id": _safe_int(rec.get("group_id"), default=""),
            "entry_price": _safe_float(rec.get("entry_price"), default=""),
            "target_price": _safe_float(rec.get("target_price"), default=""),
            "return_value": _safe_float(rec.get("return_value"), default=""),
            "reason": rec.get("reason",""),
            "context": (context or rec.get("context") or "evaluation"),
            "note": rec.get("note",""),
            "regime": rec.get("regime",""),
            "meta_choice": rec.get("meta_choice",""),
            "raw_prob": _safe_float(rec.get("raw_prob"), default=""),
            "calib_prob": _safe_float(rec.get("calib_prob"), default=""),
            "calib_ver": rec.get("calib_ver",""),
            "feature_hash": fh,
            "feature_vector": json.dumps(fv, ensure_ascii=False) if isinstance(fv,(list,tuple,np.ndarray)) else (fv or ""),
            "source": rec.get("source",""),
            "source_exchange": rec.get("source_exchange","BYBIT"),
        }

        auto_reason = _classify_failure_reason({**rec, **row})
        if not str(row["reason"]).strip():
            row["reason"] = auto_reason

        if check_failure_exists({**rec, **row}):
            return False

        try: _append_wrong_csv_row(row)
        except Exception as e: print(f"[failure_db] CSV 기록 실패: {e}")

        try:
            max_trials = 5
            for k in range(max_trials):
                try:
                    with _db_lock:
                        conn = _get_db()
                        c = conn.cursor()
                        c.execute("""
                            INSERT OR IGNORE INTO failures
                            (ts,symbol,strategy,predicted_class,label,model,group_id,reason,context,regime,raw_prob,calib_prob,feature_hash)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (row["timestamp"], row["symbol"], row["strategy"],
                              None if row["predicted_class"]=="" else row["predicted_class"],
                              None if row["label"]=="" else row["label"],
                              row["model"], None if row["group_id"]=="" else row["group_id"],
                              row["reason"], row["context"], row["regime"],
                              None if row["raw_prob"]=="" else row["raw_prob"],
                              None if row["calib_prob"]=="" else row["calib_prob"],
                              row["feature_hash"]))
                        conn.commit(); c.close()
                    break
                except sqlite3.OperationalError as oe:
                    if "locked" in str(oe).lower() and k < max_trials-1:
                        time.sleep(0.05*(k+1)); continue
                    raise
        except Exception as e:
            print(f"[failure_db] sqlite 기록 실패: {e}")

        if row["reason"] in ["negative_label","nan_label","prob_nan","class_out_of_range","bounds_mismatch"]:
            _emit_alert(f"{row['symbol']}-{row['strategy']} reason={row['reason']} pcls={row['predicted_class']} label={row['label']}")

        return True
    except Exception as e:
        print(f"[failure_db] insert_failure_record 예외: {e}")
        return False
