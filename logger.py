# === logger.py (v2025-11-05 FINAL — BASE 통합, /persistent 제거, 부팅시 파일 보장) ===
import os
import csv
import json
import datetime
import pandas as pd
import pytz
import hashlib
import shutil
import re
import sqlite3
from collections import defaultdict, deque
import threading
import time
from typing import Optional, Any, Dict
from sklearn.metrics import classification_report
from config import get_TRAIN_LOG_PATH, get_PREDICTION_LOG_PATH  # 경로 단일화

# ─────────────────────────────────────
# 0) BASE 경로 통일
#    - 이제부터는 /persistent 직접 쓰지 말고 여기로만 온다.
#    - sitecustomize가 있어도 여기서 한 번 더 안전하게 만들어준다.
# ─────────────────────────────────────
BASE = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/opt/render/project/src/persistent"
)

# 필수 폴더/파일 미리 만들어두기
try:
    os.makedirs(BASE, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    wrong_path_boot = os.path.join(BASE, "wrong_predictions.csv")
    if not os.path.exists(wrong_path_boot):
        with open(wrong_path_boot, "w", encoding="utf-8-sig") as f:
            f.write("")
except Exception:
    # 여긴 로거라 최대한 안 죽고 넘어가야 한다
    pass

# ─────────────────────────────────────
# 1) 루트/로그 경로 실제로 여기만 보게 하기
# ─────────────────────────────────────
PERSISTENT_ROOT = BASE  # ← 핵심: 예전처럼 /persistent 고정 아님
DIR = PERSISTENT_ROOT
LOG_DIR = os.path.join(DIR, "logs")
PREDICTION_LOG = get_PREDICTION_LOG_PATH()
WRONG = os.path.join(DIR, "wrong_predictions.csv")
EVAL_RESULT = os.path.join(LOG_DIR, "evaluation_result.csv")
TRAIN_LOG = get_TRAIN_LOG_PATH()
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")

# -------------------------
# 로그 레벨/샘플링 유틸
# -------------------------
LOGGER_DEBUG = os.getenv("LOGGER_DEBUG", "0") == "1"
_once_printed = set()

def _print_once(tag: str, msg: str):
    try:
        if tag not in _once_printed:
            _once_printed.add(tag); print(msg)
        elif LOGGER_DEBUG:
            print("[DBG] " + msg)
    except Exception:
        pass

_cache_hit_counts = defaultdict(int)
_CACHE_HIT_LOG_SAMPLE = max(1, int(os.getenv("CACHE_HIT_LOG_SAMPLE", "50")))
def log_cache_hit(name: str):
    try:
        _cache_hit_counts[name] += 1
        c = _cache_hit_counts[name]
        if c == 1 or (c % _CACHE_HIT_LOG_SAMPLE == 0) or LOGGER_DEBUG:
            print(f"[CACHE HIT] {name} count={c}")
    except Exception:
        pass

# -------------------------
# 경계 요약 로깅 옵션
# -------------------------
LOG_BOUNDARY_SUMMARY = os.getenv("LOG_BOUNDARY_SUMMARY", "0") == "1"
LOG_BOUNDARY_TOPK    = max(1, int(os.getenv("LOG_BOUNDARY_TOPK", "20")))
LOG_BOUNDARY_BUCKET  = float(os.getenv("LOG_BOUNDARY_BUCKET", "0.01"))

def _bucketize(v: float, step: float) -> tuple:
    try:
        import math
        base = math.floor(v / step) * step
        lo = round(base, 6); hi = round(base + step, 6)
        return (lo, hi)
    except Exception:
        return (v, v)

# -------------------------
# 파일시스템 상태 감지
# -------------------------
def _fs_has_space(path: str, min_bytes: int = 1_048_576) -> bool:
    try:
        s = os.statvfs(os.path.dirname(path) or "/")
        return (s.f_bavail * s.f_frsize) >= max(0, int(min_bytes))
    except Exception:
        return True  # 보수적으로 true

def _fs_writable(dir_path: str) -> bool:
    try:
        os.makedirs(dir_path, exist_ok=True)
        test_path = os.path.join(dir_path, ".writetest.tmp")
        with open(test_path, "w") as f:
            f.write("1")
        os.remove(test_path)
        return True
    except Exception:
        return False

# 전역 플래그: 읽기전용/용량부족 시 파일/DB 쓰기 비활성화
_READONLY_FS = not _fs_writable(LOG_DIR) or not _fs_has_space(LOG_DIR, 512*1024)
if _READONLY_FS:
    _print_once("readonlyfs", "🛑 [logger] storage read-only 또는 free space 부족 → 모든 파일/DB 쓰기 안전 강하")

# 디렉토리 생성 시도(실패해도 진행)
try:
    if not _READONLY_FS:
        os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    _READONLY_FS = True
    _print_once("mkdir_fail", f"🛑 [logger] 로그 디렉토리 생성 실패 → read-only 강하: {e}")

# -------------------------
# 공용 헤더
# -------------------------
BASE_PRED_HEADERS = [
    "timestamp","symbol","strategy","direction",
    "entry_price","target_price",
    "model","predicted_class","top_k","note",
    "success","reason","rate","return_value",
    "label","group_id","model_symbol","model_name",
    "source","volatility","source_exchange"
]
EXTRA_PRED_HEADERS = ["regime","meta_choice","raw_prob","calib_prob","calib_ver"]
CLASS_RANGE_HEADERS = ["class_return_min","class_return_max","class_return_text"]
NOTE_EXTRACT_HEADERS = ["position","hint_allow_long","hint_allow_short","hint_slope","used_minret_filter","explore_used","hint_ma_fast","hint_ma_slow"]
PREDICTION_HEADERS = BASE_PRED_HEADERS + EXTRA_PRED_HEADERS + ["feature_vector"] + CLASS_RANGE_HEADERS + NOTE_EXTRACT_HEADERS + [
    "expected_return_mid","raw_prob_pred","calib_prob_pred","meta_choice_detail"
]

TRAIN_HEADERS = [
    "timestamp","symbol","strategy","model",
    "val_acc","val_f1","val_loss",
    "engine","window","recent_cap",
    "rows","limit","min","augment_needed","enough_for_training",
    "note","source_exchange","status"
]

CHUNK = 50_000
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# -------------------------
# 간단 파일락 (read-only면 비활성)
# -------------------------
_PRED_LOCK_PATH = PREDICTION_LOG + ".lock"
_LOCK_STALE_SEC = 120

class _FileLock:
    def __init__(self, path: str, timeout: float = 10.0, poll: float = 0.05):
        self.path = path; self.timeout = float(timeout); self.poll = float(poll)
    def __enter__(self):
        if _READONLY_FS:  # 잠금 불필요
            return self
        deadline = time.time() + self.timeout
        while True:
            try:
                if os.path.exists(self.path):
                    try:
                        mtime = os.path.getmtime(self.path)
                        if (time.time() - mtime) > _LOCK_STALE_SEC:
                            try: os.remove(self.path)
                            except Exception: pass
                    except Exception: pass
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} ts={time.time()}\n")
                break
            except FileExistsError:
                if time.time() >= deadline:
                    try:
                        mtime = os.path.getmtime(self.path)
                        if (time.time() - mtime) > _LOCK_STALE_SEC:
                            try: os.remove(self.path); continue
                            except Exception: pass
                    except Exception: pass
                    raise TimeoutError(f"lock timeout: {self.path}")
                time.sleep(self.poll)
        return self
    def __exit__(self, exc_type, exc, tb):
        if _READONLY_FS: return
        try:
            if os.path.exists(self.path): os.remove(self.path)
        except Exception: pass

# -------------------------
# 연속 실패 집계기
# -------------------------
class _ConsecutiveFailAggregator:
    TH = max(2, int(os.getenv("FAIL_SUMMARY_THRESHOLD", "5")))
    WINDOW = max(60, int(os.getenv("FAIL_SUMMARY_WINDOW", "900")))
    _state = defaultdict(lambda: {"cnt":0, "first_ts":0.0, "last_ts":0.0, "last_reason":""})

    @classmethod
    def _flush(cls, key, where="periodic"):
        st = cls._state.get(key)
        if not st or st["cnt"] <= 0: return
        sym, strat, gid, model = key
        msg = f"[연속실패요약/{where}] {sym}-{strat}-g{gid} {model} ×{st['cnt']} (last_reason={st['last_reason']})"
        if not _READONLY_FS:
            try:
                os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
                with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=["timestamp","symbol","strategy","status","reason"])
                    if f.tell() == 0: w.writeheader()
                    w.writerow({
                        "timestamp": now_kst().isoformat(),
                        "symbol": sym, "strategy": strat,
                        "status": f"consecutive_fail_{st['cnt']}",
                        "reason": st["last_reason"]
                    })
            except Exception:
                pass
        _print_once(f"cfail:{sym}:{strat}:{gid}:{model}", "🔻 " + msg)
        cls._state.pop(key, None)

    @classmethod
    def add(cls, key, success: bool, reason: str = ""):
        now = time.time()
        st = cls._state[key]
        if st["last_ts"] and now - st["last_ts"] > cls.WINDOW:
            cls._flush(key, "stale"); st = cls._state[key]
        if success:
            if st["cnt"] > 0: cls._flush(key, "recovered")
            cls._state.pop(key, None); return
        st["cnt"] = int(st["cnt"]) + 1
        st["last_reason"] = (reason or "")[:200]; st["last_ts"] = now
        if st["first_ts"] == 0.0: st["first_ts"] = now
        if st["cnt"] % cls.TH == 0: cls._flush(key, "periodic")

# -------------------------
# 안전한 로그 파일 보장
# -------------------------
def _read_csv_header(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first = f.readline().strip()
        if not first: return []
        return [h.strip() for h in first.split(",")]
    except Exception:
        return []

def ensure_prediction_log_exists():
    if _READONLY_FS: return
    try:
        os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)
        if not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0:
            with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(PREDICTION_HEADERS)
            print("[✅ ensure_prediction_log_exists] prediction_log.csv 생성(확장 스키마)")
        else:
            existing = _read_csv_header(PREDICTION_LOG)
            if existing != PREDICTION_HEADERS:
                bak = PREDICTION_LOG + ".bak"
                try: os.replace(PREDICTION_LOG, bak)
                except Exception:
                    try: shutil.copyfile(PREDICTION_LOG, bak); open(PREDICTION_LOG, "w", encoding="utf-8-sig").close()
                    except Exception: return
                with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as out, \
                     open(bak, "r", encoding="utf-8-sig") as src:
                    w = csv.writer(out); w.writerow(PREDICTION_HEADERS)
                    reader = csv.reader(src)
                    try: next(reader)
                    except StopIteration: reader = []
                    for row in reader:
                        row = (row + [""] * len(PREDICTION_HEADERS))[:len(PREDICTION_HEADERS)]
                        w.writerow(row)
                print("[✅ ensure_prediction_log_exists] 기존 파일 헤더 보정(확장) 완료")
    except Exception as e:
        print(f"[⚠️ ensure_prediction_log_exists] 예외: {e}")

def ensure_train_log_exists():
    if _READONLY_FS: return
    try:
        os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)
        if not os.path.exists(TRAIN_LOG) or os.path.getsize(TRAIN_LOG) == 0:
            with open(TRAIN_LOG, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(TRAIN_HEADERS)
            print("[✅ ensure_train_log_exists] train_log.csv 생성(확장 스키마)")
        else:
            existing = _read_csv_header(TRAIN_LOG)
            if existing != TRAIN_HEADERS:
                bak = TRAIN_LOG + ".bak"
                try: os.replace(TRAIN_LOG, bak)
                except Exception:
                    try: shutil.copyfile(TRAIN_LOG, bak); open(TRAIN_LOG, "w", encoding="utf-8-sig").close()
                    except Exception: return
                with open(TRAIN_LOG, "w", newline="", encoding="utf-8-sig") as out, \
                     open(bak, "r", encoding="utf-8-sig") as src:
                    w = csv.writer(out); w.writerow(TRAIN_HEADERS)
                    reader = csv.reader(src)
                    try: old_header = next(reader)
                    except StopIteration: old_header = []
                    for row in reader:
                        mapped = {h:row[i] for i,h in enumerate(old_header)} if old_header else {}
                        val_loss_val = mapped.get("val_loss", mapped.get("loss", mapped.get("train_loss_sum","")))
                        new_row = [
                            mapped.get("timestamp",""), mapped.get("symbol",""), mapped.get("strategy",""), mapped.get("model",""),
                            mapped.get("accuracy", mapped.get("val_acc","")), mapped.get("f1", mapped.get("val_f1","")), val_loss_val,
                            "", "", "", "", "", "", "", "",
                            mapped.get("note",""), mapped.get("source_exchange",""), mapped.get("status",""),
                        ]
                        w.writerow(new_row[:len(TRAIN_HEADERS)])
                print("[✅ ensure_train_log_exists] train_log.csv 헤더 보정(확장) 완료")
    except Exception as e:
        print(f"[⚠️ ensure_train_log_exists] 예외: {e}")

# -------------------------
# 로그 로테이션 (읽기전용이면 skip)
# -------------------------
def rotate_prediction_log_if_needed(max_mb: int = 200, backups: int = 3):
    if _READONLY_FS: return
    try:
        if not os.path.exists(PREDICTION_LOG): return
        size_mb = os.path.getsize(PREDICTION_LOG) / (1024 * 1024)
        if size_mb < max_mb: return
        for i in range(backups, 0, -1):
            dst = f"{PREDICTION_LOG}.{i}"
            src = f"{PREDICTION_LOG}.{i-1}" if i-1 > 0 else PREDICTION_LOG
            if os.path.exists(src):
                try:
                    if os.path.exists(dst): os.remove(dst)
                    shutil.move(src, dst)
                except Exception:
                    pass
        ensure_prediction_log_exists()
        print(f"[logger] 🔁 rotate: {size_mb:.1}MB → rotated with {backups} backups")
    except Exception as e:
        print(f"[logger] rotate error: {e}")

# -------------------------
# feature hash
# -------------------------
def get_feature_hash(feature_row) -> str:
    try:
        import numpy as _np
        if feature_row is None: return "none"
        if "torch" in str(type(feature_row)):
            try: feature_row = feature_row.detach().cpu().numpy()
            except Exception: pass
        if isinstance(feature_row, _np.ndarray):
            arr = feature_row.flatten().astype(float)
        elif isinstance(feature_row, (list, tuple)):
            arr = _np.array(feature_row, dtype=float).flatten()
        else:
            arr = _np.array([float(feature_row)], dtype=float)
        rounded = [round(float(x), 2) for x in arr]
        joined = ",".join(map(str, rounded))
        return hashlib.sha1(joined.encode()).hexdigest()
    except Exception:
        return "hash_error"

# -------------------------
# SQLite: 모델 성공/실패 집계 (I/O 에러 시 메모리로 강하)
# -------------------------
_db_conn = None
_DB_PATH = os.path.join(LOG_DIR, "failure_patterns.db")
_db_lock = threading.RLock()
_DB_ENABLED = os.getenv("LOG_DISABLE_SQL", "0") != "1" and not _READONLY_FS

def _apply_sqlite_pragmas(conn):
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.close()
    except Exception as e:
        print(f"[경고] PRAGMA 설정 실패 → {e}")

def _connect_sqlite():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=30, check_same_thread=False)
    _apply_sqlite_pragmas(conn)
    return conn

def get_db_connection():
    global _db_conn, _DB_ENABLED
    if not _DB_ENABLED:
        return None
    with _db_lock:
        if _db_conn is None:
            try:
                _db_conn = _connect_sqlite()
                try:
                    cur = _db_conn.cursor(); cur.execute("SELECT 1;"); cur.close()
                except Exception:
                    try: _db_conn.close()
                    except Exception: pass
                    _db_conn = _connect_sqlite()
                print("[✅ logger.py DB connection 생성 완료]")
            except Exception as e:
                print(f"[오류] logger.py DB connection 생성 실패 → {e}")
                _db_conn = None
                _DB_ENABLED = False
        return _db_conn

def _sqlite_exec_with_retry(sql, params=(), retries=5, sleep_base=0.2, commit=False):
    if not _DB_ENABLED:
        raise sqlite3.OperationalError("db disabled")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with _db_lock:
                conn = get_db_connection()
                if conn is None:
                    raise sqlite3.OperationalError("db unavailable")
                cur = conn.cursor()
                cur.execute(sql, params)
                if commit: conn.commit()
                try: rows = cur.fetchall()
                except sqlite3.ProgrammingError: rows = None
                cur.close()
                return rows
        except sqlite3.OperationalError as e:
            msg = str(e).lower(); last_err = e
            transient = ("database is locked" in msg) or ("disk i/o error" in msg) or ("database is busy" in msg)
            if transient:
                if "disk i/o error" in msg or "no space left" in msg:
                    print("[🛑 logger.db] disk I/O 오류 감지 → DB 기능 비활성화")
                    globals()["_DB_ENABLED"] = False
                    return None
                time.sleep(sleep_base * attempt); continue
            else:
                raise
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * attempt); continue
    raise last_err if last_err else RuntimeError("sqlite exec failed")

def ensure_success_db():
    if not _DB_ENABLED: return
    try:
        _sqlite_exec_with_retry("""
            CREATE TABLE IF NOT EXISTS model_success (
                symbol TEXT,
                strategy TEXT,
                model TEXT,
                success INTEGER,
                fail INTEGER,
                PRIMARY KEY(symbol, strategy, model)
            )
        """, params=(), retries=5, commit=True)
    except Exception as e:
        print(f"[오류] ensure_success_db 실패 → {e}")
        globals()["_DB_ENABLED"] = False

def update_model_success(s, t, m, success):
    if not _DB_ENABLED:
        _print_once("db_disabled_warn", "ℹ️ model_success 집계는 현재 메모리/콘솔만 기록")
        return
    try:
        _sqlite_exec_with_retry("""
            INSERT INTO model_success (symbol, strategy, model, success, fail)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, strategy, model) DO UPDATE SET
                success = success + excluded.success,
                fail = fail  + excluded.fail
        """, params=(s, t or "알수없음", m, int(success), int(0 if success else 1)), retries=7, commit=True)
        print(f"[✅ update_model_success] {s}-{t}-{m} 기록 ({'성공' if success else '실패'})")
    except Exception as e:
        print(f"[오류] update_model_success 실패 → {e}")
        globals()["_DB_ENABLED"] = False

def get_model_success_rate(s, t, m):
    if not _DB_ENABLED: return 0.0
    try:
        rows = _sqlite_exec_with_retry("""
            SELECT success, fail FROM model_success
            WHERE symbol=? AND strategy=? AND model=?
        """, params=(s, t or "알수없음", m), retries=5, commit=False)
        row = rows[0] if rows else None
        if row is None: return 0.0
        success_cnt, fail_cnt = row; total = success_cnt + fail_cnt
        return (success_cnt / total) if total > 0 else 0.0
    except Exception as e:
        print(f"[오류] get_model_success_rate 실패] {e}"); return 0.0

# -------------------------
# failure_db 초기화
# -------------------------
try:
    from failure_db import ensure_failure_db as _ensure_failure_db_once
    if not _READONLY_FS:
        _ensure_failure_db_once()
    print("[logger] failure_db initialized (schema ready)")
except Exception as _e:
    print(f"[logger] failure_db init failed: {_e}")

# 서버 시작 시 보장
ensure_success_db()
ensure_prediction_log_exists()
ensure_train_log_exists()

# -------------------------
# 파일 로드/유틸
# -------------------------
def load_failure_count():
    path = os.path.join(LOG_DIR, "failure_count.csv")
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}
    except:
        return {}

def _normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower().map(lambda x: "success" if x == "success" else "fail"); return df
    if "success" in df.columns:
        s = df["success"].map(lambda x: str(x).strip().lower() in ["true","1","yes","y"])
        df["status"] = s.map(lambda b: "success" if b else "fail"); return df
    df["status"] = ""; return df

# -------------------------
# 메모리 안전 집계
# -------------------------
def get_meta_success_rate(strategy, min_samples: int = 1):
    if not os.path.exists(PREDICTION_LOG): return 0.0
    usecols = ["timestamp","strategy","model","status","success"]
    succ = total = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success"]],
        chunksize=CHUNK
    ):
        if "model" in chunk.columns:
            chunk = chunk[chunk["model"] == "meta"]
        if "strategy" in chunk.columns:
            chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty: continue
        if "status" in chunk.columns and chunk["status"].notna().any():
            mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
            chunk = chunk[mask]
            s = chunk["status"].astype(str).str.lower().isin(["success","v_success"])
            succ += int(s.sum()); total += int(len(chunk))
        elif "success" in chunk.columns:
            s = chunk["success"].astype(str).str.lower().isin(["true","1","success","v_success"])
            succ += int(s.sum()); total += int(len(chunk))
    if total < max(1, min_samples): return 0.0
    return float(succ / total)

def get_strategy_eval_count(strategy: str):
    if not os.path.exists(PREDICTION_LOG): return 0
    usecols = ["strategy","status","success"]
    count = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success"]],
        chunksize=CHUNK
    ):
        if "strategy" in chunk.columns:
            chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty: continue
        if "status" in chunk.columns and chunk["status"].notna().any():
            mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
            count += int(mask.sum())
        elif "success" in chunk.columns:
            count += int(len(chunk))
    return int(count)

def get_actual_success_rate(strategy, min_samples: int = 1):
    if not os.path.exists(PREDICTION_LOG): return 0.0
    usecols = ["strategy","status","success"]
    succ = total = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success"]],
        chunksize=CHUNK
    ):
        if "strategy" in chunk.columns:
            chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty: continue
        if "status" in chunk.columns and chunk["status"].notna().any():
            mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
            s = chunk["status"].astype(str).str.lower().isin(["success","v_success"])
            succ += int(s[mask].sum()); total += int(mask.sum())
        elif "success" in chunk.columns:
            s = chunk["success"].astype(str).str.lower().isin(["true","1","success","v_success"])
            succ += int(s.sum()); total += int(len(chunk))
    if total < max(1, min_samples): return 0.0
    return round(succ / total, 4)

# -------------------------
# 감사 로그
# -------------------------
def log_audit_prediction(s, t, status, reason):
    row = {
        "timestamp": now_kst().isoformat(),
        "symbol": str(s or "UNKNOWN"),
        "strategy": str(t or "알수없음"),
        "status": str(status),
        "reason": str(reason)
    }
    if _READONLY_FS:
        print(f"[AUDIT][console] {json.dumps(row, ensure_ascii=False)}"); return
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0: w.writeheader()
            w.writerow(row)
    except:
        pass

# -------------------------
# 예측 로그
# -------------------------
def _align_row_to_header(row, header):
    if len(row) < len(header): row = row + [""] * (len(header) - len(row))
    elif len(row) > len(header): row = row[:len(header)]
    return row

def _clean_str(x):
    s = str(x).strip() if x is not None else ""
    return "" if s.lower() in {"","unknown","none","nan","null"} else s

def _normalize_model_fields(model, model_name, symbol, strategy):
    m = _clean_str(model); mn = _clean_str(model_name)
    if not m and mn: m = mn
    if not mn and m: mn = m
    if not m and not mn:
        base = f"auto_{symbol}_{strategy}"; m = mn = base
    return m, mn

def _extract_from_note(note_str: str):
    fields = {
        "position": "", "hint_allow_long": "", "hint_allow_short": "", "hint_slope": "",
        "used_minret_filter": "", "explore_used": "", "hint_ma_fast": "", "hint_ma_slow": ""
    }
    try:
        obj = json.loads(note_str) if isinstance(note_str, str) else {}
        if isinstance(obj, dict):
            for k in list(fields.keys()):
                if k in obj:
                    v = obj.get(k)
                    if isinstance(v, bool): fields[k] = int(v)
                    else: fields[k] = v if (v is not None) else ""
            if not fields.get("explore_used"):
                mc = obj.get("meta_choice", ""); fields["explore_used"] = 1 if ("explore" in str(mc)) else 0
    except Exception:
        pass
    return fields

def log_prediction(
    symbol, strategy, direction=None, entry_price=0, target_price=0,
    timestamp=None, model=None, predicted_class=None, top_k=None,
    note="", success=False, reason="", rate=None, return_value=None,
    label=None, group_id=None, model_symbol=None, model_name=None,
    source="일반", volatility=False, feature_vector=None,
    source_exchange="BYBIT", regime=None, meta_choice=None,
    raw_prob=None, calib_prob=None, calib_ver=None,
    class_return_min=None, class_return_max=None, class_return_text=None,
    expected_return=None,
    **kwargs
):
    from datetime import datetime as _dt
    if not _READONLY_FS:
        ensure_prediction_log_exists()

    if rate is None:
        rate = expected_return if expected_return is not None else 0.0

    now = _dt.now(pytz.timezone("Asia/Seoul")).isoformat() if timestamp is None else timestamp
    top_k_str = ",".join(map(str, top_k)) if top_k else ""
    reason = (reason or "").strip()
    rate = float(rate)
    return_value = 0.0 if return_value is None else float(return_value)
    entry_price = float(entry_price or 0.0)
    target_price = float(target_price or 0.0)
    model, model_name = _normalize_model_fields(model, model_name, symbol, strategy)

    try:
        note_obj = json.loads(note) if isinstance(note, str) else {}
    except Exception:
        note_obj = {}
    expected_return_mid = note_obj.get("expected_return_mid", "")
    raw_prob_pred = note_obj.get("raw_prob_pred", "")
    calib_prob_pred = note_obj.get("calib_prob_pred", "")
    meta_choice_detail = note_obj.get("meta_choice", "")

    fv_serial = ""
    try:
        if feature_vector is not None:
            import numpy as np
            if isinstance(feature_vector, np.ndarray): v = feature_vector.flatten().tolist()
            elif isinstance(feature_vector, (list, tuple)): v = feature_vector
            else: v = []
            fv_serial = json.dumps(v if len(v) <= 64 else {"head": v[:8], "tail": v[-8:]}, ensure_ascii=False)
    except Exception:
        fv_serial = ""

    # note에서 추가 필드 뽑기
    note_ex = _extract_from_note(note)

    row = [
        now, symbol, strategy, direction, entry_price, target_price,
        model, predicted_class, top_k_str, note, str(success), reason,
        rate, return_value, label, group_id, model_symbol, model_name,
        source, volatility, source_exchange, regime, meta_choice,
        raw_prob, calib_prob, calib_ver, fv_serial,
        class_return_min, class_return_max, class_return_text,
        note_ex.get("position",""), note_ex.get("hint_allow_long",""), note_ex.get("hint_allow_short",""),
        note_ex.get("hint_slope",""), note_ex.get("used_minret_filter",""), note_ex.get("explore_used",""),
        note_ex.get("hint_ma_fast",""), note_ex.get("hint_ma_slow",""),
        expected_return_mid, raw_prob_pred, calib_prob_pred, meta_choice_detail
    ]

    if _READONLY_FS or not _fs_has_space(PREDICTION_LOG, 256*1024):
        payload = dict(zip(PREDICTION_HEADERS, _align_row_to_header(row, PREDICTION_HEADERS)))
        print(f"[PREDICT][console] {json.dumps(payload, ensure_ascii=False)}")
    else:
        with _FileLock(_PRED_LOCK_PATH, timeout=10.0):
            rotate_prediction_log_if_needed()
            write_header = not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0
            with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if write_header: w.writerow(PREDICTION_HEADERS)
                w.writerow(_align_row_to_header(row, PREDICTION_HEADERS))

    if success:
        _print_once(f"pred_ok:{symbol}:{strategy}:{model_name}",
                    f"[✅ 예측 OK] {symbol}-{strategy} class={predicted_class} rate={rate:.4f} src={source_exchange}")
    else:
        _ConsecutiveFailAggregator.add((symbol, strategy, group_id or 0, model_name), False, reason)

# -------------------------
# 학습 로그
# -------------------------
_note_re_engine   = re.compile(r"engine=([a-zA-Z_]+)")
_note_re_window   = re.compile(r"window=(\d+)")
_note_re_cap      = re.compile(r"cap=(\d+)")
_note_re_flags    = re.compile(r"data_flags=\{?rows:(\d+),\s*limit:(\d+),\s*min:(\d+),\s*aug:(\d+),\s*enough_for_training:(\d+)\}?")

def _parse_train_note(note: str):
    s = str(note or "")
    eng = (_note_re_engine.search(s) or [None, ""])[1]
    win = (_note_re_window.search(s) or [None, ""])[1]
    cap = (_note_re_cap.search(s) or [None, ""])[1]
    mfl = _note_re_flags.search(s)
    rows = limit = minv = aug = enough = ""
    if mfl:
        rows, limit, minv, aug, enough = mfl.groups()
    return {
        "engine": eng, "window": win, "recent_cap": cap,
        "rows": rows, "limit": limit, "min": minv,
        "augment_needed": aug, "enough_for_training": enough
    }

def _first_non_none(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None

def log_training_result(
    symbol, strategy, model="",
    accuracy=None, f1=None, loss=None,
    note="", source_exchange="BYBIT", status="success",
    y_true=None, y_pred=None, num_classes=None,
    **kwargs
):
    LOG_FILE = TRAIN_LOG
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()

    extras = _parse_train_note(note)

    try:
        val_acc  = _first_non_none(kwargs.get("val_acc"), accuracy)
        val_f1   = _first_non_none(kwargs.get("val_f1"),  f1)
        val_loss = _first_non_none(kwargs.get("val_loss"), loss)
        val_acc  = float(val_acc)  if val_acc  is not None and str(val_acc)  != "" else 0.0
        val_f1   = float(val_f1)   if val_f1   is not None and str(val_f1)   != "" else 0.0
        val_loss = float(val_loss) if val_loss is not None and str(val_loss) != "" else 0.0
    except Exception:
        val_acc, val_f1, val_loss = 0.0, 0.0, 0.0

    row = [
        now, str(symbol), str(strategy), str(model or ""),
        val_acc, val_f1, val_loss,
        extras.get("engine",""), extras.get("window",""), extras.get("recent_cap",""),
        extras.get("rows",""), extras.get("limit",""), extras.get("min",""),
        extras.get("augment_needed",""), extras.get("enough_for_training",""),
        str(note or ""), str(source_exchange or "BYBIT"),
        str(status or "success")
    ]
    try:
        if _READONLY_FS:
            print(f"[TRAIN][console] {json.dumps(dict(zip(TRAIN_HEADERS, _align_row_to_header(row, TRAIN_HEADERS))), ensure_ascii=False)}")
        else:
            write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
            if write_header: ensure_train_log_exists()
            with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if write_header: w.writerow(TRAIN_HEADERS)
                w.writerow(_align_row_to_header(row, TRAIN_HEADERS))
        _f1_key = (str(symbol), str(strategy))
        if not hasattr(log_training_result, "_f1_zero"):
            log_training_result._f1_zero = defaultdict(int)
        if float(val_f1 or 0.0) <= 0.0:
            log_training_result._f1_zero[_f1_key] += 1
            n = log_training_result._f1_zero[_f1_key]
            if n == 1:
                print(f"🟠 [경고] F1=0.0 발생 → {symbol}-{strategy} {model} (1회)")
            elif n % int(os.getenv('F1_ZERO_WARN_EVERY','5')) == 0:
                print(f"🟠 [요약] F1=0.0 연속 {n}회 → {symbol}-{strategy} {model}")
        else:
            if getattr(log_training_result, "_f1_zero", {}).get(_f1_key, 0) > 0:
                print(f"[✅ 복구] {symbol}-{strategy} {model} F1 회복 → {float(val_f1 or 0.0):.4f}")
            log_training_result._f1_zero[_f1_key] = 0
        _print_once(f"trainlog:{symbol}:{strategy}:{model}",
                    f"[✅ 학습 로그 기록] {symbol}-{strategy} {model} val_f1={float(val_f1 or 0.0):.4f} status={status}")
    except Exception as e:
        print(f"[⚠️ 학습 로그 기록 실패] {e}")

    try:
        if not _READONLY_FS and os.path.exists(LOG_FILE):
            N = int(os.getenv("LOG_F1_MA_N", "20"))
            df_ma = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
            if "val_f1" in df_ma.columns:
                sub = df_ma[df_ma.get("strategy","") == strategy].tail(max(1, N))
                if not sub.empty:
                    ma_f1 = float(pd.to_numeric(sub["val_f1"], errors="coerce").dropna().mean()) if "val_f1" in sub else float("nan")
                    if ma_f1 == ma_f1:
                        print(f"[📊 이동평균 F1] 전략={strategy} 최근{len(sub)}회 → {ma_f1:.4f}")
    except Exception:
        pass

    if y_true is not None and y_pred is not None:
        try:
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            per_cls = {k: v["f1-score"] for k, v in rep.items() if k.isdigit()}
            print(f"[📊 per-class F1] {symbol}-{strategy} {model} → {json.dumps(per_cls, ensure_ascii=False)}")
            counts = {int(k): int(v["support"]) for k, v in rep.items() if k.isdigit()}
            log_eval_coverage(symbol=symbol, strategy=strategy, counts=counts,
                              num_classes=(num_classes if num_classes is not None else len(per_cls)),
                              note="train_end")
        except Exception as e:
            print(f"[⚠️ per-class F1/coverage 계산 실패] {e}")

# -------------------------
# 수익률 클래스 경계 로그
# -------------------------
def log_class_ranges(symbol, strategy, group_id=None, class_ranges=None, note=""):
    import csv, os
    path = os.path.join(LOG_DIR, "class_ranges.csv")
    now = now_kst().isoformat()
    class_ranges = class_ranges or []

    if LOG_BOUNDARY_SUMMARY and class_ranges:
        bucket = max(LOG_BOUNDARY_BUCKET, 1e-9); agg = {}
        for rng in class_ranges:
            try:
                lo, hi = float(rng[0]), float(rng[1]); mid = (lo + hi) / 2.0
            except Exception: continue
            blo, bhi = _bucketize(mid, bucket)
            d = agg.setdefault((blo, bhi), {"cnt":0, "min":lo, "max":hi, "sum_mid":0.0})
            d["cnt"] += 1; d["min"] = min(d["min"], lo); d["max"] = max(d["max"], hi); d["sum_mid"] += mid

        rows = []
        for (blo, bhi), d in agg.items():
            mean_mid = d["sum_mid"] / max(1, d["cnt"])
            rows.append({
                "timestamp": now, "symbol": str(symbol), "strategy": str(strategy),
                "group_id": int(group_id) if group_id is not None else 0,
                "bucket_lo": float(blo), "bucket_hi": float(bhi),
                "count": int(d["cnt"]), "min": float(d["min"]), "max": float(d["max"]),
                "mean_mid": float(round(mean_mid, 6)), "note": str(note or "")
            })
        rows.sort(key=lambda r: (r["count"], abs(r["mean_mid"])), reverse=True)
        top_rows = rows[:LOG_BOUNDARY_TOPK]

        sum_path = os.path.join(LOG_DIR, "class_ranges_summary.csv")
        if _READONLY_FS:
            print(f"[CLASSRANGE_SUM][console] {json.dumps({'rows':top_rows}, ensure_ascii=False)}")
        else:
            write_header = not os.path.exists(sum_path)
            try:
                with open(sum_path, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "timestamp","symbol","strategy","group_id",
                        "bucket_lo","bucket_hi","count","min","max","mean_mid","note"
                    ])
                    if write_header: w.writeheader()
                    for r in top_rows: w.writerow(r)
                _print_once(f"class_ranges_sum:{symbol}:{strategy}",
                            f"[📐 클래스경계 요약] {symbol}-{strategy}-g{group_id} → buckets={len(rows)} topk={len(top_rows)} (step={bucket})")
            except Exception as e:
                print(f"[⚠️ 클래스경계 요약 로그 실패] {e}")

        if not _READONLY_FS:
            write_header_detail = not os.path.exists(path)
            try:
                with open(path, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    if write_header_detail: w.writerow(["timestamp","symbol","strategy","group_id","idx","low","high","note"])
                    w.writerow([now, symbol, strategy, int(group_id) if group_id is not None else 0, -1, "", "", f"summary_only step={bucket} topk={LOG_BOUNDARY_TOPK}"])
            except Exception as e:
                print(f"[⚠️ 클래스경계(요약마커) 기록 실패] {e}")
        return

    if _READONLY_FS:
        print(f"[CLASSRANGE][console] {json.dumps({'timestamp':now,'symbol':symbol,'strategy':strategy,'group_id':group_id,'ranges':class_ranges}, ensure_ascii=False)}")
        return

    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","idx","low","high","note"])
            for i, rng in enumerate(class_ranges):
                try: lo, hi = (float(rng[0]), float(rng[1]))
                except Exception: lo, hi = (None, None)
                w.writerow([now, symbol, strategy, int(group_id) if group_id is not None else 0, i, lo, hi, str(note or "")])
        _print_once(f"class_ranges:{symbol}:{strategy}", f"[📐 클래스경계 로그] {symbol}-{strategy}-g{group_id} → {len(class_ranges)}개 기록")
    except Exception as e:
        print(f"[⚠️ 클래스경계 로그 실패] {e}")

# -------------------------
# 수익률 분포 요약 로그
# -------------------------
def log_return_distribution(symbol, strategy, group_id=None, horizon_hours=None, summary: dict=None, note=""):
    path = os.path.join(LOG_DIR, "return_distribution.csv")
    now = now_kst().isoformat()
    s = summary or {}
    row = [
        now, str(symbol), str(strategy),
        int(group_id) if group_id is not None else 0,
        int(horizon_hours) if horizon_hours is not None else "",
        float(s.get("min", 0.0)), float(s.get("p25", 0.0)), float(s.get("p50", 0.0)),
        float(s.get("p75", 0.0)), float(s.get("p90", 0.0)), float(s.get("p95", 0.0)),
        float(s.get("p99", 0.0)), float(s.get("max", 0.0)), int(s.get("count", 0)),
        str(note or "")
    ]
    if _READONLY_FS:
        print(f"[RETDIST][console] {json.dumps(dict(zip(['timestamp','symbol','strategy','group_id','horizon_hours','min','p25','p50','p75','p90','p95','p99','max','count','note'], row)), ensure_ascii=False)}")
        return
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","horizon_hours",
                            "min","p25","p50","p75","p90","p95","p99","max","count","note"])
            w.writerow(row)
        _print_once(f"ret_dist:{symbol}:{strategy}", f"[📈 수익률분포 로그] {symbol}-{strategy}-g{group_id} count={s.get('count',0)}")
    except Exception as e:
        print(f"[⚠️ 수익률분포 로그 실패] {e}")

# -------------------------
# 라벨 분포 로그
# -------------------------
def log_label_distribution(symbol, strategy, group_id=None, counts: dict=None, total: int=None, n_unique: int=None, entropy: float=None, labels=None, note=""):
    import json, math
    path = os.path.join(LOG_DIR, "label_distribution.csv")
    now = now_kst().isoformat()

    if counts is None:
        from collections import Counter
        try: labels_list = list(map(int, list(labels or [])))
        except Exception: labels_list = []
        cnt = Counter(labels_list); total_calc = sum(cnt.values())
        probs = [c/total_calc for c in cnt.values()] if total_calc > 0 else []
        entropy_calc = -sum(p*math.log(p + 1e-12) for p in probs) if probs else 0.0
        counts = {int(k): int(v) for k, v in sorted(cnt.items())}
        total = total_calc; n_unique = len(cnt); entropy = round(float(entropy_calc), 6)
    else:
        counts = {int(k): int(v) for k, v in sorted((counts or {}).items())}
        total = int(total if total is not None else sum(counts.values()))
        n_unique = int(n_unique if n_unique is not None else len(counts))
        if entropy is None:
            probs = [c/total for c in counts.values()] if total > 0 else []
            entropy = round(float(-sum(p*math.log(p + 1e-12) for p in probs)) if probs else 0.0, 6)
        else:
            entropy = float(entropy)

    row = [
        now, str(symbol), str(strategy),
        int(group_id) if group_id is not None else 0,
        int(total), json.dumps(counts, ensure_ascii=False),
        int(n_unique), float(entropy), str(note or "")
    ]

    if _READONLY_FS:
        print(f"[LABELDIST][console] {json.dumps(dict(zip(['timestamp','symbol','strategy','group_id','total','counts_json','n_unique','entropy','note'], row)), ensure_ascii=False)}")
        return

    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","total","counts_json","n_unique","entropy","note"])
            w.writerow(row)
        _print_once(f"label_dist:{symbol}:{strategy}", f"[📊 라벨분포 로그] {symbol}-{strategy}-g{group_id} → total={total}, classes={n_unique}, H={entropy:.4f}")
    except Exception as e:
        print(f"[⚠️ 라벨분포 로그 실패] {e}")

# -------------------------
# 검증 커버리지 로그
# -------------------------
def log_eval_coverage(symbol: str, strategy: str, counts: dict, num_classes: int, note: str = ""):
    path = os.path.join(LOG_DIR, "validation_coverage.csv")
    now = now_kst().isoformat()
    counts = {int(k): int(v) for k, v in sorted((counts or {}).items())}
    covered = sum(1 for v in counts.values() if int(v) > 0)
    total = int(sum(counts.values()))
    coverage = (covered / max(1, int(num_classes))) if num_classes else 0.0

    if _READONLY_FS:
        out = {
            "timestamp": now, "symbol": symbol, "strategy": strategy,
            "num_classes": int(num_classes), "covered": int(covered),
            "coverage": float(round(coverage,4)), "total": int(total),
            "counts_json": json.dumps(counts, ensure_ascii=False), "note": str(note or "")
        }
        print(f"[VALCOV][console] {json.dumps(out, ensure_ascii=False)}")
    else:
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                fieldnames = ["timestamp","symbol","strategy","num_classes","covered","coverage","total","counts_json","note"]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header: w.writeheader()
                w.writerow({
                    "timestamp": now, "symbol": symbol, "strategy": strategy,
                    "num_classes": int(num_classes), "covered": int(covered),
                    "coverage": float(round(coverage,4)), "total": int(total),
                    "counts_json": json.dumps(counts, ensure_ascii=False), "note": str(note or "")
                })
        except Exception as e:
            print(f"[⚠️ validation_coverage 로그 실패] {e}")

    if covered <= 1:
        print(f"🔴 [경고] 검증 라벨 단일 클래스 감지 → {symbol}-{strategy} (covered={covered}/{num_classes})")
    elif coverage < 0.6:
        print(f"🟠 [주의] 검증 클래스 커버 낮음 → {symbol}-{strategy} (coverage={coverage:.2f})")

def alert_if_single_class_prediction(symbol: str, strategy: str, lookback_days: int = 3, min_rows: int = 100):
    try:
        if _READONLY_FS or not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0:
            return False
        cutoff = now_kst() - datetime.timedelta(days=int(lookback_days))
        uniq = set(); total = 0
        usecols = ["timestamp","symbol","strategy","predicted_class"]
        for chunk in pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", usecols=[c for c in usecols if c in PREDICTION_HEADERS], chunksize=CHUNK):
            if "timestamp" not in chunk.columns or "predicted_class" not in chunk.columns: continue
            ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
            try: ts = ts.dt.tz_localize("Asia/Seoul")
            except Exception:
                try: ts = ts.dt.tz_convert("Asia/Seoul")
                except Exception: pass
            sub = chunk[(chunk.get("symbol","")==symbol) & (chunk.get("strategy","")==strategy)]
            if sub.empty: continue
            ts_sub = ts.loc[sub.index]; sub = sub.loc[ts_sub >= cutoff]
            pcs = pd.to_numeric(sub["predicted_class"], errors="coerce").dropna().astype(int)
            uniq.update(pcs.unique().tolist()); total += int(len(pcs))
        if total >= int(min_rows) and len(uniq) <= 1:
            print(f"🔴 [경고] 최근 예측이 사실상 단일 클래스 → {symbol}-{strategy} (rows={total}, uniq={len(uniq)})")
            log_audit_prediction(symbol, strategy, "single_class_pred", f"rows={total}, uniq={len(uniq)}")
            return True
        return False
    except Exception as e:
        print(f"[⚠️ 단일클래스 예측 점검 실패] {e}")
        return False

# -------------------------
# 정렬 키
# -------------------------
def _model_sort_key(r):
    return (str(r.get("symbol","")), str(r.get("strategy","")), str(r.get("model","")), int(r.get("group_id",0)))

# -------------------------
# 모델 인벤토리 조회
# -------------------------
def get_available_models(symbol: str = None, strategy: str = None):
    try:
        model_dir = os.path.join(PERSISTENT_ROOT, "models")
        if not os.path.isdir(model_dir): return []
        out = []; exts = (".pt", ".ptz", ".safetensors")
        def _stem_meta(path: str) -> str:
            b = os.path.basename(path)
            for e in exts:
                if b.endswith(e):
                    return os.path.join(os.path.dirname(path), b[: -len(e)] + ".meta.json")
            return path + ".meta.json"
        for fn in os.listdir(model_dir):
            if not fn.endswith(exts): continue
            pt_path = os.path.join(model_dir, fn); meta_path = _stem_meta(pt_path)
            if not os.path.exists(meta_path): continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
            except Exception: meta = {}
            sym = meta.get("symbol") or fn.split("_", 1)[0]
            strat = meta.get("strategy") or ("단기" if "_단기_" in fn else "중기" if "_중기_" in fn else "장기" if "_장기_" in fn else "")
            if symbol and sym != symbol: continue
            if strategy and strat != strategy: continue
            out.append({
                "pt_file": fn, "meta_file": os.path.basename(meta_path),
                "symbol": sym, "strategy": strat, "model": meta.get("model",""),
                "group_id": meta.get("group_id",0), "num_classes": meta.get("num_classes",0),
                "val_f1": float(meta.get("metrics", {}).get("val_f1", 0.0)),
                "timestamp": meta.get("timestamp","")
            })
        out.sort(key=_model_sort_key); return out
    except Exception as e:
        print(f"[오류] get_available_models 실패 → {e}"); return []

# -------------------------
# 최근 예측 통계 내보내기
# -------------------------
def export_recent_model_stats(days: int = 7, out_path: str = None):
    try:
        if out_path is None:
            os.makedirs(LOG_DIR, exist_ok=True)
            out_path = os.path.join(LOG_DIR, "recent_model_stats.csv")
        if not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0:
            pd.DataFrame(columns=["symbol","strategy","model","total","success","fail","success_rate","last_ts"]).to_csv(out_path, index=False, encoding="utf-8-sig")
            return out_path
        cutoff = now_kst() - datetime.timedelta(days=int(days))
        agg = {}
        usecols = ["timestamp","symbol","strategy","model","status","success"]
        for chunk in pd.read_csv(
            PREDICTION_LOG, encoding="utf-8-sig",
            usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success"]],
            chunksize=CHUNK
        ):
            if "timestamp" in chunk.columns:
                ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
                try: ts = ts.dt.tz_localize("Asia/Seoul")
                except Exception:
                    try: ts = ts.dt.tz_convert("Asia/Seoul")
                    except Exception: pass
                chunk = chunk.loc[ts >= cutoff]; chunk = chunk.assign(_ts=ts)
            else: continue
            if chunk.empty or "model" not in chunk.columns: continue
            succ_mask = None
            if "status" in chunk.columns and chunk["status"].notna().any():
                ok_mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
                succ_mask = chunk["status"].astype(str).str.lower().isin(["success","v_success"])
                chunk = chunk[ok_mask]
            elif "success" in chunk.columns:
                succ_mask = chunk["success"].astype(str).str.lower().isin(["true","1","success","v_success"])
            else: continue
            if chunk.empty: continue
            chunk = chunk.assign(_succ=succ_mask.astype(bool))
            for (sym, strat, mdl), sub in chunk.groupby(["symbol","strategy","model"], dropna=False):
                key = (str(sym), str(strat), str(mdl))
                d = agg.setdefault(key, {"success":0, "total":0, "last_ts":None})
                d["success"] += int(sub["_succ"].sum()); d["total"] += int(len(sub))
                last_ts = pd.to_datetime(sub["_ts"].max(), errors="coerce")
                if d["last_ts"] is None or (pd.notna(last_ts) and last_ts > d["last_ts"]):
                    d["last_ts"] = last_ts
        rows = []
        for (sym,strat,mdl), d in agg.items():
            total = int(d["total"]); succ = int(d["success"])
            rate = (succ/total) if total>0 else 0.0
            last_ts = d["last_ts"].isoformat() if d["last_ts"] is not None else ""
            rows.append({"symbol": sym, "strategy": strat, "model": mdl, "total": total, "success": succ, "fail": total - succ, "success_rate": round(rate, 4), "last_ts": last_ts})
        df_out = pd.DataFrame(rows, columns=["symbol","strategy","model","total","success","fail","success_rate","last_ts"])
        df_out = df_out.sort_values(["symbol","strategy","model","last_ts"])
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[✅ export_recent_model_stats] 저장: {out_path} (rows={len(df_out)})")
        return out_path
    except Exception as e:
        print(f"[⚠️ export_recent_model_stats 실패] {e}")
        try:
            if out_path is None: out_path = os.path.join(LOG_DIR, "recent_model_stats.csv")
            pd.DataFrame(columns=["symbol","strategy","model","total","success","fail","success_rate","last_ts"]).to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception: pass
        return out_path or os.path.join(LOG_DIR, "recent_model_stats.csv")

# ============================================================
# 관우 요약 로그 생성기 — group_trigger 포함
# ============================================================
def flush_gwanwoo_summary():
    """
    관우(시각화) 로그용 summary.csv 생성.
    자동예측(group_trigger 등) 포함. 저장 불가 시 콘솔 폴백.
    """
    from config import get_GANWU_PATH, get_PREDICTION_LOG_PATH
    gw_dir = get_GANWU_PATH()                         # /data/guanwu/incoming

    # --- (1) 경로 통합: 평가/예측 경로 정확화 ---
    # 평가 결과는 시스템 표준 로그 위치(/persistent/logs → 지금은 BASE/logs)를 사용
    paths = {
        "pred_json": os.path.join(gw_dir, "prediction_result.json"),
        "eval_csv": EVAL_RESULT,  # 표준 로그 경로에서 읽음
    }

    # 예측 로그는 존재하는 첫 후보를 사용
    pred_csv_candidates = [
        PREDICTION_LOG,
        os.path.join(gw_dir, "prediction_log.csv"),
        get_PREDICTION_LOG_PATH()
    ]
    pred_csv_path = next((p for p in pred_csv_candidates if isinstance(p, str) and os.path.exists(p)), pred_csv_candidates[0])
    paths["pred_csv"] = pred_csv_path

    out_path = os.path.join(gw_dir, "gwanwoo_summary.csv")
    records = []

    # 1) prediction_result.json
    try:
        if os.path.exists(paths["pred_json"]):
            with open(paths["pred_json"], "r", encoding="utf-8") as f:
                js = json.load(f)
            if isinstance(js, list):
                for j in js:
                    records.append({
                        "timestamp": j.get("timestamp",""),
                        "symbol": j.get("symbol",""),
                        "strategy": j.get("strategy",""),
                        "predicted_class": j.get("predicted_class",""),
                        "expected_return": j.get("expected_return",""),
                        "prob": j.get("prob",""),
                        "model": j.get("model","meta"),
                        "source": j.get("source","prediction_result")
                    })
    except Exception as e:
        print(f"[⚠️ 관우요약] prediction_result.json 읽기 실패: {e}")

    # 2) evaluation_result.csv
    try:
        if os.path.exists(paths["eval_csv"]):
            df = pd.read_csv(paths["eval_csv"], encoding="utf-8-sig")
            if not df.empty:
                df["source"] = "evaluation"
                records.extend(df.to_dict("records"))
    except Exception as e:
        print(f"[⚠️ 관우요약] evaluation_result.csv 읽기 실패: {e}")

    # 3) prediction_log.csv
    try:
        if os.path.exists(paths["pred_csv"]):
            df = pd.read_csv(paths["pred_csv"], encoding="utf-8-sig")
            if not df.empty:
                src_col = "source" if "source" in df.columns else None
                if src_col:
                    blacklist = {"debug","dry_run"}
                    df = df[~df[src_col].astype(str).isin(blacklist)]
                keep = [c for c in ["timestamp","symbol","strategy","predicted_class",
                                    "rate","raw_prob","calib_prob","success","reason","source"] if c in df.columns]
                if keep:
                    records.extend(df[keep].to_dict("records"))
                else:
                    records.extend(df.to_dict("records"))
    except Exception as e:
        print(f"[⚠️ 관우요약] prediction_log.csv 읽기 실패: {e}")

    if not records:
        msg_hdr = ["timestamp","symbol","strategy","predicted_class","expected_return","prob","model","source"]
        if _READONLY_FS or not _fs_has_space(out_path, 256*1024):
            print(f"[GWANWOO_SUM][console] []")
            return out_path
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(msg_hdr)
        except Exception as e:
            print(f"[⚠️ 관우요약] summary 헤더 생성 실패: {e}")
        return out_path

    df_out = pd.DataFrame(records)
    for col in ["expected_return","prob","model","predicted_class","source"]:
        if col not in df_out.columns: df_out[col] = ""
    df_out = df_out.fillna("")

    if _READONLY_FS or not _fs_has_space(out_path, 256*1024):
        print(f"[GWANWOO_SUM][console] {df_out.to_json(orient='records', force_ascii=False)}")
    else:
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        except Exception:
            pass
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[✅ 관우요약] {len(df_out)}행 생성: {out_path}")
    return out_path
