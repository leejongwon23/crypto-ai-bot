# === logger.py (v2025-11-05 + meta/hold 확장본 + QUIET MODE 추가) ===
import sitecustomize
import os
import csv
import json
import datetime
import pandas as pd
import numpy as np
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

# ==========================================================
# ✅ 운영로그 너무 많을 때만 "조용히" 만드는 옵션
# - LOGGER_MODE=quiet  → logger.py 내부 print 대부분 숨김(경고/오류/핵심만 표시)
# - 기본값(normal)     → 예전처럼 그대로 출력
# ==========================================================
import builtins as _builtins

LOGGER_MODE = os.getenv("LOGGER_MODE", "normal").strip().lower()  # normal | quiet

# quiet 모드에서 "살릴" 로그 키워드(핵심/경고/오류/완료)
_QUIET_KEEP_TOKENS = (
    "🛑", "⚠️", "🔴", "🟠",  # 위험/경고
    "[오류]", "[경고]",      # 한글 태그
    "[✅", "✅",             # 완료/성공
    "failure_db init failed",
    "storage read-only",
    "free space 부족",
    "single_class",
    "LABEL_SINGLE_CLASS",
    "STATUS_FAIL",
    "F1_ZERO",
)

def _logger_print(*args, **kwargs):
    """
    logger.py 내부 print만 조용히 필터링.
    - normal: 그대로 출력
    - quiet : 중요한 것만 출력
    """
    try:
        if LOGGER_MODE != "quiet":
            return _builtins.print(*args, **kwargs)

        msg = " ".join([str(a) for a in args])
        # 너무 긴 JSON payload는 quiet에서 기본 숨김 (경고성/에러성 아니면)
        if any(tok in msg for tok in _QUIET_KEEP_TOKENS):
            return _builtins.print(*args, **kwargs)

        # 기본은 숨김
        return
    except Exception:
        # print 자체가 죽으면 로거가 더 위험해지니 그냥 무시
        return

# ✅ 이 파일(logger.py) 안에서 호출되는 print는 전부 여기로 들어옴
print = _logger_print

# ─────────────────────────────────────
# 0) BASE 경로 통일
# ─────────────────────────────────────
BASE = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/opt/render/project/src/persistent"
)

# 디렉터리까지만 만들고, 파일은 여기서 "절대" 안 만든다
try:
    os.makedirs(BASE, exist_ok=True)
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
except Exception:
    # 로거는 최대한 안 죽고 넘어가야 한다
    pass

# ─────────────────────────────────────
# 1) 루트/로그 경로 실제로 여기만 보게 하기
# ─────────────────────────────────────
PERSISTENT_ROOT = BASE  # ← 핵심: 예전처럼 /persistent 고정 아님
DIR = PERSISTENT_ROOT
LOG_DIR = os.path.join(DIR, "logs")
PREDICTION_LOG = get_PREDICTION_LOG_PATH()
WRONG = os.path.join(DIR, "wrong_predictions.csv")  # 실제로는 쓰는 쪽에서 만들 것
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

# 🔹 로그 분석 시 무시할 source 값들(훈련용/디버그)
LOG_SOURCE_BLACKLIST = {
    "debug",
    "dry_run",
    "train",
    "train_return_distribution",
    "train_dist",
}

# -------------------------
# 파일시스템 상태 감지
# -------------------------
def _fs_has_space(path: str, min_bytes: int = 1_048_576) -> bool:
    try:
        base = os.path.dirname(path) or "/"
        s = os.statvfs(base)
        return (s.f_bavail * s.f_frsize) >= max(0, int(min_bytes))
    except Exception:
        return True

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

# ✅ 각 로그 파일이 "실제로 쓰이는 위치" 기준으로 read-only 판단
_LOGDIR = LOG_DIR
_TRAINDIR = os.path.dirname(TRAIN_LOG) if isinstance(TRAIN_LOG, str) else LOG_DIR
_PREDDIR  = os.path.dirname(PREDICTION_LOG) if isinstance(PREDICTION_LOG, str) else LOG_DIR

_READONLY_LOGDIR  = (not _fs_writable(_LOGDIR))  or (not _fs_has_space(_LOGDIR,  512*1024))
_READONLY_TRAIN   = (not _fs_writable(_TRAINDIR)) or (not _fs_has_space(_TRAINDIR, 512*1024))
_READONLY_PRED    = (not _fs_writable(_PREDDIR))  or (not _fs_has_space(_PREDDIR,  512*1024))

# ✅ 기존 호환: "전체 read-only"는 세 개 중 하나라도 막히면 True
_READONLY_FS = _READONLY_LOGDIR or _READONLY_TRAIN or _READONLY_PRED

if _READONLY_FS:
    _print_once(
        "readonlyfs",
        "🛑 [logger] storage read-only 또는 free space 부족 → 일부/전체 파일 쓰기 차단"
        f" (LOG_DIR={int(_READONLY_LOGDIR)}, TRAIN_DIR={int(_READONLY_TRAIN)}, PRED_DIR={int(_READONLY_PRED)})"
    )

# 디렉토리 생성 시도(실패해도 진행)
try:
    if not _READONLY_LOGDIR:
        os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    _READONLY_LOGDIR = True
    _READONLY_FS = True
    _print_once("mkdir_fail", f"🛑 [logger] LOG_DIR 생성 실패 → read-only 강하: {e}")


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

# 🔹 여기서 메타/섀도우/보류용 헤더를 추가
META_HOLD_HEADERS = [
    "expected_return_mid","raw_prob_pred","calib_prob_pred","meta_choice_detail",
    "chosen_model","chosen_class","shadow_models","shadow_classes",
    "hold_type","hold_reason","meta_score","meta_reason",
]

PREDICTION_HEADERS = (
    BASE_PRED_HEADERS
    + EXTRA_PRED_HEADERS
    + ["feature_vector"]
    + CLASS_RANGE_HEADERS
    + NOTE_EXTRACT_HEADERS
    + META_HOLD_HEADERS
)

# 🔹 학습 로그 기본 헤더 + 수익률/클래스 확장 헤더 분리
TRAIN_BASE_HEADERS = [
    "timestamp","symbol","strategy","model",
    "val_acc","val_f1","val_loss",
    "engine","window","recent_cap",
    "rows","limit","min","augment_needed","enough_for_training",
    "note","source_exchange","status"
]

TRAIN_EXTRA_HEADERS = [
    "class_edges","class_counts","class_ranges",
    "bin_spans","near_zero_band","near_zero_count",
    "NUM_CLASSES","usable_samples","masked_count",
    "per_class_f1",
]

# 🔹 실제 CSV에서 사용할 전체 헤더
TRAIN_HEADERS = TRAIN_BASE_HEADERS + TRAIN_EXTRA_HEADERS

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
    if _READONLY_FS:
        return
    try:
        os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)

        # 새로 만들기 또는 빈 파일이면 헤더 생성
        if not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0:
            with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(PREDICTION_HEADERS)
            print("[✅ ensure_prediction_log_exists] prediction_log.csv 생성(확장 스키마)")
        else:
            # 기존 헤더 확인 후 다르면 보정
            existing = _read_csv_header(PREDICTION_LOG)
            if existing != PREDICTION_HEADERS:
                bak = PREDICTION_LOG + ".bak"

                # 안전 백업
                try:
                    os.replace(PREDICTION_LOG, bak)
                except Exception:
                    try:
                        shutil.copyfile(PREDICTION_LOG, bak)
                        open(PREDICTION_LOG, "w", encoding="utf-8-sig").close()
                    except Exception:
                        return

                with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as out, \
                     open(bak, "r", encoding="utf-8-sig") as src:

                    w = csv.writer(out)
                    w.writerow(PREDICTION_HEADERS)

                    reader = csv.reader(src)
                    try:
                        next(reader)  # 기존 헤더 스킵
                    except StopIteration:
                        reader = []

                    # 기존 데이터 재적재
                    for row in reader:
                        row = (row + [""] * len(PREDICTION_HEADERS))[:len(PREDICTION_HEADERS)]
                        w.writerow(row)

                print("[✅ ensure_prediction_log_exists] 기존 파일 헤더 보정(확장) 완료")

    except Exception as e:
        print(f"[⚠️ ensure_prediction_log_exists] 예외: {e}")


def ensure_train_log_exists():
    """
    ✅ 핵심 목표
    - train_log.csv 가 없으면: 최신 TRAIN_HEADERS로 생성
    - 헤더가 다르면: 백업 후 "기존 데이터 최대 보존"하면서 헤더만 업그레이드
      (절대 빈칸으로 밀어버리지 않음)
    """
    # ✅ 핵심: 환경/권한/디스크 상황이 런타임에 바뀌면 매번 재판정해야 함
    try:
        _refresh_fs_flags()
    except Exception:
        pass

    # ✅ 추가: read-only면 절대 건드리지 않음
    if _READONLY_TRAIN:
        return

    try:
        os.makedirs(os.path.dirname(TRAIN_LOG), exist_ok=True)

        # 1) 없으면 생성
        if not os.path.exists(TRAIN_LOG) or os.path.getsize(TRAIN_LOG) == 0:
            with open(TRAIN_LOG, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=TRAIN_HEADERS)
                w.writeheader()
            print("[✅ ensure_train_log_exists] train_log.csv 생성(확장 스키마)")
            return

        existing = _read_csv_header(TRAIN_LOG)
        if existing == TRAIN_HEADERS:
            return  # 이미 정상이면 아무것도 안 함

        # 2) 헤더가 다르면: 백업 + 보존 업그레이드
        bak = TRAIN_LOG + ".bak"
        try:
            os.replace(TRAIN_LOG, bak)
        except Exception:
            shutil.copyfile(TRAIN_LOG, bak)

        with open(bak, "r", encoding="utf-8-sig", newline="") as src, \
             open(TRAIN_LOG, "w", encoding="utf-8-sig", newline="") as out:

            reader = csv.DictReader(src)

            writer = csv.DictWriter(out, fieldnames=TRAIN_HEADERS)
            writer.writeheader()

            for old_row in reader:
                old_row = old_row or {}

                # ✅ 구버전 키 호환 매핑 (있으면 살려서 넣기)
                if (not old_row.get("val_acc")) and old_row.get("accuracy") not in (None, ""):
                    old_row["val_acc"] = old_row.get("accuracy")
                if (not old_row.get("val_f1")) and old_row.get("f1") not in (None, ""):
                    old_row["val_f1"] = old_row.get("f1")
                if (not old_row.get("val_loss")):
                    v = old_row.get("val_loss")
                    if v in (None, ""):
                        v = old_row.get("loss")
                    if v in (None, ""):
                        v = old_row.get("train_loss_sum")
                    if v not in (None, ""):
                        old_row["val_loss"] = v

                new_row = {}
                for h in TRAIN_HEADERS:
                    if h in old_row and old_row[h] not in (None, ""):
                        new_row[h] = old_row[h]
                    else:
                        new_row[h] = ""

                writer.writerow(new_row)

        print(f"[train_log] 헤더 업그레이드 완료 → backup={bak}")

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
        print(f"[logger] 🔁 rotate: {size_mb:.1f}MB → rotated with {backups} backups")
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

def _refresh_fs_flags():
    global _READONLY_LOGDIR, _READONLY_TRAIN, _READONLY_PRED, _READONLY_FS

    _LOGDIR = LOG_DIR
    _TRAINDIR = os.path.dirname(TRAIN_LOG) if isinstance(TRAIN_LOG, str) and TRAIN_LOG else LOG_DIR
    _PREDDIR  = os.path.dirname(PREDICTION_LOG) if isinstance(PREDICTION_LOG, str) and PREDICTION_LOG else LOG_DIR

    _READONLY_LOGDIR  = (not _fs_writable(_LOGDIR))   or (not _fs_has_space(_LOGDIR,  512*1024))
    _READONLY_TRAIN   = (not _fs_writable(_TRAINDIR)) or (not _fs_has_space(_TRAINDIR, 512*1024))
    _READONLY_PRED    = (not _fs_writable(_PREDDIR))  or (not _fs_has_space(_PREDDIR,  512*1024))

    _READONLY_FS = _READONLY_LOGDIR or _READONLY_TRAIN or _READONLY_PRED
        
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
        print(f"[오류] update_model_success 실패] {e}")
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
        try:
            _ensure_failure_db_once()
        except FileNotFoundError as fe:
            if "wrong_predictions.csv" in str(fe):
                print("[logger] failure_db init skipped (missing wrong_predictions.csv — caller에서 생성돼야 함)")
            else:
                raise
    print("[logger] failure_db initialized (schema ready]")
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
    usecols = ["timestamp","strategy","model","status","success","source"]
    succ = total = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success","source"]],
        chunksize=CHUNK
    ):
        if "source" in chunk.columns:
            chunk = chunk[~chunk["source"].astype(str).isin(LOG_SOURCE_BLACKLIST)]
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
    usecols = ["strategy","status","success","source"]
    count = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success","source"]],
        chunksize=CHUNK
    ):
        if "source" in chunk.columns:
            chunk = chunk[~chunk["source"].astype(str).isin(LOG_SOURCE_BLACKLIST)]
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
    usecols = ["strategy","status","success","source"]
    succ = total = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success","source"]],
        chunksize=CHUNK
    ):
        if "source" in chunk.columns:
            chunk = chunk[~chunk["source"].astype(str).isin(LOG_SOURCE_BLACKLIST)]
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
    chosen_model=None, chosen_class=None,
    shadow_models=None, shadow_classes=None,
    hold_type=None, hold_reason=None,
    meta_score=None,
    meta_reason=None,
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
    meta_reason = (meta_reason or "").strip()
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

    note_ex = _extract_from_note(note)

    if isinstance(shadow_models, (list, tuple, set)):
        shadow_models_str = "|".join(map(str, shadow_models))
    else:
        shadow_models_str = "" if shadow_models is None else str(shadow_models)

    if isinstance(shadow_classes, (list, tuple, set)):
        shadow_classes_str = "|".join(map(str, shadow_classes))
    else:
        shadow_classes_str = "" if shadow_classes is None else str(shadow_classes)

    src_lower = str(source or "").lower()
    mdl_lower = str(model or "").lower()
    rsn_lower = str(reason or "").lower()
    is_train_dist = (
        "train" in src_lower
        or mdl_lower == "trainer"
        or rsn_lower == "train_return_distribution"
    )

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
        expected_return_mid, raw_prob_pred, calib_prob_pred, meta_choice_detail,
        chosen_model or "", chosen_class if chosen_class is not None else "",
        shadow_models_str, shadow_classes_str,
        hold_type or "", hold_reason or "", meta_score if meta_score is not None else "",
        meta_reason,
    ]

    aligned = _align_row_to_header(row, PREDICTION_HEADERS)
    payload = dict(zip(PREDICTION_HEADERS, aligned))

    if _READONLY_FS or not _fs_has_space(PREDICTION_LOG, 256*1024):
        tag = "PREDICT/TRAIN_DIST" if is_train_dist else "PREDICT"
        print(f"[{tag}][console] {json.dumps(payload, ensure_ascii=False)}")
    else:
        if is_train_dist:
            print(f"[PREDICT/TRAIN_DIST][console] {json.dumps(payload, ensure_ascii=False)}")
        else:
            with _FileLock(_PRED_LOCK_PATH, timeout=10.0):
                rotate_prediction_log_if_needed()
                write_header = not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0
                with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    if write_header: w.writerow(PREDICTION_HEADERS)
                    w.writerow(aligned)

    if success:
        if is_train_dist:
            _print_once(
                f"train_ret_dist:{symbol}:{strategy}:{model_name}",
                f"[✅ 수익분포 로그] {symbol}-{strategy} ({model_name}) — 학습용 수익률 분포만 기록 (실제 매매 예측 아님)"
            )
        else:
            _print_once(
                f"pred_ok:{symbol}:{strategy}:{model_name}",
                f"[✅ 예측 OK] {symbol}-{strategy} class={predicted_class} rate={rate:.4f} src={source_exchange}"
            )
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

    m = _note_re_engine.search(s)
    eng = m.group(1) if m else ""

    m = _note_re_window.search(s)
    win = m.group(1) if m else ""

    m = _note_re_cap.search(s)
    cap = m.group(1) if m else ""

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
    class_edges=None,
    class_counts=None,
    class_ranges=None,
    bin_spans=None,
    near_zero_band=None,
    near_zero_count=None,
    per_class_f1=None,
    masked_count=None,
    NUM_CLASSES=None,
    usable_samples=None,
    **kwargs
):
    """
    ✅ 학습 로그의 '진실성'을 보장하는 단일 진입점
    - 성능이 계산 안 되면 절대 success 로 기록하지 않음
    - 단일 클래스면 무조건 status=fail
    - F1=0 은 명확히 기록
    """

    # ✅ 핵심: 런타임에 writable 상태가 바뀔 수 있으니 매번 갱신
    try:
        _refresh_fs_flags()
    except Exception:
        pass

    # ✅ 추가: read-only면 절대 기록 시도하지 않음
    if _READONLY_TRAIN:
        return

    # ✅ 추가 가드: 깨진/디버그성 모델명은 train_log에 기록 금지
    mlow = str(model or "").strip().lower()
    if mlow in {"all", "trainer", "nan", "none", "null"}:
        return

    LOG_FILE = TRAIN_LOG
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()

    extras = _parse_train_note(note)

    def _sf(x):
        try:
            if x in [None, "", "nan", "NaN", "None", "null"]:
                return None
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def _si(x):
        try:
            if x in [None, "", "nan", "NaN", "None", "null"]:
                return None
            return int(float(x))
        except Exception:
            return None

    # -------------------------
    # 1) 성능 값 정규화
    # -------------------------
    val_acc  = _sf(kwargs.get("val_acc", accuracy))
    val_f1   = _sf(kwargs.get("val_f1",  f1))
    val_loss = _sf(kwargs.get("val_loss", loss))

    # -------------------------
    # 2) 클래스 수 판단 (가장 중요)
    # -------------------------
    real_num_classes = (
        _si(NUM_CLASSES)
        or _si(num_classes)
        or (len(class_counts) if isinstance(class_counts, (list, dict)) else None)
    )

    # -------------------------
    # 3) 상태(status) 강제 판정
    # -------------------------
    final_status = str(status or "").lower()

    # (A) 성능 자체가 없으면 실패
    if val_acc is None or val_f1 is None:
        final_status = "fail"

    # (B) 단일 클래스면 무조건 실패
    if real_num_classes is not None and real_num_classes <= 1:
        final_status = "fail"

    # (C) F1 = 0 은 실패
    if val_f1 is not None and val_f1 <= 0.0:
        final_status = "fail"

    if final_status not in {"success", "ok"}:
        final_status = "fail"

    # -------------------------
    # 4) usable_samples 보정
    # -------------------------
    if usable_samples is None:
        usable_samples = _si(extras.get("rows")) or 0

    # -------------------------
    # 5) 로그 row 구성
    # -------------------------
    row = {
        "timestamp": now,
        "symbol": str(symbol),
        "strategy": str(strategy),
        "model": str(model or ""),

        "val_acc": 0.0 if val_acc is None else float(val_acc),
        "val_f1":  0.0 if val_f1  is None else float(val_f1),
        "val_loss": "" if val_loss is None else float(val_loss),

        "engine": extras.get("engine",""),
        "window": extras.get("window",""),
        "recent_cap": extras.get("recent_cap",""),

        "rows": extras.get("rows",""),
        "limit": extras.get("limit",""),
        "min": extras.get("min",""),
        "augment_needed": extras.get("augment_needed",""),
        "enough_for_training": extras.get("enough_for_training",""),

        "note": str(note or ""),
        "source_exchange": str(source_exchange or "BYBIT"),
        "status": final_status,

        "class_edges": json.dumps(class_edges or [], ensure_ascii=False),
        "class_counts": json.dumps(class_counts or [], ensure_ascii=False),
        "class_ranges": json.dumps(class_ranges or [], ensure_ascii=False),
        "bin_spans": json.dumps(bin_spans or [], ensure_ascii=False),

        "near_zero_band": _sf(near_zero_band) or 0.0,
        "near_zero_count": _si(near_zero_count) or 0,

        "NUM_CLASSES": _si(real_num_classes) or 0,
        "usable_samples": int(usable_samples),

        "per_class_f1": json.dumps(per_class_f1 or [], ensure_ascii=False),
        "masked_count": _si(masked_count) or 0,
    }

    # -------------------------
    # 6) 기록
    # -------------------------
    try:
        ensure_train_log_exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=TRAIN_HEADERS, extrasaction="ignore")
            w.writerow(row)

        _print_once(
            f"trainlog:{symbol}:{strategy}:{model}",
            f"[📘 학습기록] {symbol}-{strategy} {model} "
            f"acc={row['val_acc']} f1={row['val_f1']} status={final_status}"
        )

        update_train_dashboard(symbol, strategy, model)

    except Exception as e:
        print(f"[🛑 학습 로그 기록 실패] {e}")

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
        usecols = ["timestamp","symbol","strategy","predicted_class","source"]
        for chunk in pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig", usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["source"]], chunksize=CHUNK):
            # source 필터
            if "source" in chunk.columns:
                chunk = chunk[~chunk["source"].astype(str).isin(LOG_SOURCE_BLACKLIST)]
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

def _safe_read_df(path: str):
    """CSV를 안전하게 읽고, 실패하면 빈 DataFrame 반환."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()

def _get_last_row(df: pd.DataFrame, filt: dict):
    """
    DataFrame 에서 filt 조건(symbol/strategy/model 등)에 맞는
    '마지막 한 줄'을 dict 로 돌려준다. 없으면 None.
    """
    if df is None or df.empty:
        return None
    for k, v in filt.items():
        if k not in df.columns:
            return None
        df = df[df[k] == v]
    if df.empty:
        return None
    return df.tail(1).to_dict("records")[0]


def update_train_dashboard(symbol: str, strategy: str, model: str = ""):
    """
    모든 학습 관련 정보를 통합한 1줄 요약 레코드를 생성한다.
    - 거짓표시 방지:
      (1) NUM_CLASSES와 class_edges 길이가 안 맞으면 edges 텍스트를 잘라서 일치시킴
      (2) return_distribution count=0이면 min/max/p50을 0으로 '가짜 표시'하지 않도록 비움 처리
      (3) label_classes<=1이면 class_ranges_text를 무조건 비움
    """
    symbol = str(symbol)
    strategy = str(strategy)
    model = str(model or "")

    if _READONLY_FS:
        return

    out_path = os.path.join(LOG_DIR, "train_dashboard.csv")

    # 1) train_log에서 최신 한 줄 찾기
    train_df = _safe_read_df(TRAIN_LOG)
    if train_df is None or train_df.empty:
        return

    required_cols = {"symbol", "strategy", "timestamp"}
    if not required_cols.issubset(train_df.columns):
        return

    sub = train_df[(train_df["symbol"] == symbol) & (train_df["strategy"] == strategy)]
    if sub.empty:
        return

    sub = sub.copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.dropna(subset=["timestamp"])
    sub = sub.sort_values("timestamp")

    # ✅ 1) 깨진/디버그성 model 제거 (model=all 같은 것 방지)
    BAD_MODELS = {"all", "trainer", "none", "nan", "null", ""}
    if "model" in sub.columns:
        sub["_m"] = sub["model"].astype(str).str.strip().str.lower()
        sub = sub[~sub["_m"].isin(BAD_MODELS)].copy()

    if sub.empty:
        return

    # ✅ 2) 성공(success/ok) 행이 있으면 “성공 중 가장 최신” 우선
    if "status" in sub.columns:
        st = sub["status"].astype(str).str.lower()
        ok = sub[st.isin(["success", "ok"])].copy()
        if not ok.empty:
            trow = ok.tail(1).to_dict("records")[0]
        else:
            trow = sub.tail(1).to_dict("records")[0]
    else:
        trow = sub.tail(1).to_dict("records")[0]

    # -----------------------
    # 유틸
    # -----------------------
    def _f(x, default=None):
        try:
            if x in ["", None, "nan", "NaN", "None", "null"]:
                return default
            v = float(x)
            if not np.isfinite(v):
                return default
            return v
        except Exception:
            return default

    def _i(x, default=None):
        try:
            if x in ["", None, "nan", "NaN", "None", "null"]:
                return default
            return int(float(x))
        except Exception:
            return default

    # 2) 기본 메트릭
    val_acc  = _f(trow.get("val_acc"), None)
    val_f1   = _f(trow.get("val_f1"), None)
    val_loss = _f(trow.get("val_loss"), None)

    # 실제 학습 클래스 수(가장 믿을만한 후보)
    real_num_classes = _i(trow.get("NUM_CLASSES"), None)
    if real_num_classes is None:
        real_num_classes = _i(trow.get("num_classes"), None)

    # -----------------------
    # 3) 라벨 분포(label_distribution.csv)
    # -----------------------
    label_total = None
    label_classes = None
    label_counts_json = ""
    label_entropy = None

    label_df = _safe_read_df(os.path.join(LOG_DIR, "label_distribution.csv"))
    lrow = None
    if not label_df.empty and {"symbol", "strategy"}.issubset(label_df.columns):
        lrow = _get_last_row(label_df, {"symbol": symbol, "strategy": strategy})

    if lrow:
        label_total = _i(lrow.get("total"), None)
        label_classes = _i(lrow.get("n_unique"), None)
        label_counts_json = str(lrow.get("counts_json", "") or "")
        label_entropy = _f(lrow.get("entropy"), None)

    # fallback: train_log의 class_counts
    if label_classes is None or label_total is None:
        counts_raw = trow.get("class_counts", "")
        counts = {}
        try:
            counts_parsed = json.loads(counts_raw) if isinstance(counts_raw, str) else counts_raw
            if isinstance(counts_parsed, list):
                counts = {i: int(c) for i, c in enumerate(counts_parsed)}
            elif isinstance(counts_parsed, dict):
                counts = {int(k): int(v) for k, v in counts_parsed.items()}
        except Exception:
            counts = {}

        if counts:
            label_total = int(sum(counts.values()))
            label_classes = int(sum(1 for v in counts.values() if int(v) > 0))
            label_counts_json = json.dumps(counts, ensure_ascii=False)

    if label_total is None:
        label_total = _i(trow.get("usable_samples"), 0) or 0
    if label_classes is None:
        label_classes = 0

    # -----------------------
    # 4) 클래스 구간 텍스트 (거짓표시 방지 핵심)
    # -----------------------
    class_ranges_text = ""

    # (A) class_ranges.csv가 있으면 그걸 쓰되,
    #     real_num_classes가 있으면 그 개수만큼만 잘라서 사용
    class_ranges_df = _safe_read_df(os.path.join(LOG_DIR, "class_ranges.csv"))
    cr_sub = pd.DataFrame()
    if not class_ranges_df.empty and {"symbol", "strategy", "idx", "low", "high"}.issubset(class_ranges_df.columns):
        cr_sub = class_ranges_df[
            (class_ranges_df["symbol"] == symbol) & (class_ranges_df["strategy"] == strategy)
        ].copy()

    if not cr_sub.empty:
        cr_sub["idx"] = pd.to_numeric(cr_sub["idx"], errors="coerce")
        cr_sub = cr_sub[cr_sub["idx"] >= 0].sort_values("idx")
        if real_num_classes is not None and real_num_classes > 0:
            cr_sub = cr_sub.head(int(real_num_classes))

        parts = []
        for _, r in cr_sub.iterrows():
            try:
                lo, hi = float(r["low"]), float(r["high"])
                idx = int(r["idx"]) + 1
                parts.append(f"C{idx}: {lo*100:.2f}% ~ {hi*100:.2f}%")
            except Exception:
                continue
        class_ranges_text = " | ".join(parts)

    # (B) fallback: class_edges
    if not class_ranges_text:
        edges_raw = trow.get("class_edges", "")
        try:
            edges = json.loads(edges_raw) if isinstance(edges_raw, str) else edges_raw
            if isinstance(edges, list) and len(edges) >= 2:
                # real_num_classes가 있으면 edges를 그 개수+1까지만 자른다
                if real_num_classes is not None and real_num_classes > 0:
                    need = int(real_num_classes) + 1
                    if len(edges) >= need:
                        edges = edges[:need]

                parts = []
                for i in range(len(edges) - 1):
                    lo, hi = float(edges[i]), float(edges[i + 1])
                    parts.append(f"C{i+1}: {lo*100:.2f}% ~ {hi*100:.2f}%")
                class_ranges_text = " | ".join(parts)
        except Exception:
            pass

    # ✅ 라벨이 1개 이하이면 "구간 텍스트"는 무조건 비움
    if int(label_classes) <= 1:
        class_ranges_text = ""

    # -----------------------
    # 4.5) 수익률 분포(return_distribution.csv) - 거짓 0.00 표시 방지
    # -----------------------
    ret_min = ret_p25 = ret_p50 = ret_p75 = ret_p90 = ret_p95 = ret_p99 = ret_max = None
    ret_count = 0

    ret_df = _safe_read_df(os.path.join(LOG_DIR, "return_distribution.csv"))
    if not ret_df.empty and {"symbol", "strategy"}.issubset(ret_df.columns):
        rsub = ret_df[(ret_df["symbol"] == symbol) & (ret_df["strategy"] == strategy)]
        if not rsub.empty:
            rlast = rsub.tail(1).iloc[0]
            ret_count = _i(rlast.get("count"), 0) or 0

            # count가 0이면 값은 비워둔다(0.00%로 가짜 표시 금지)
            if ret_count > 0:
                ret_min = _f(rlast.get("min"), None)
                ret_p25 = _f(rlast.get("p25"), None)
                ret_p50 = _f(rlast.get("p50"), None)
                ret_p75 = _f(rlast.get("p75"), None)
                ret_p90 = _f(rlast.get("p90"), None)
                ret_p95 = _f(rlast.get("p95"), None)
                ret_p99 = _f(rlast.get("p99"), None)
                ret_max = _f(rlast.get("max"), None)

    # -----------------------
    # 4.6) 검증 커버리지(validation_coverage.csv)
    # -----------------------
    val_num_classes = val_covered = 0
    val_coverage = 0.0

    cov_df = _safe_read_df(os.path.join(LOG_DIR, "validation_coverage.csv"))
    if not cov_df.empty and {"symbol", "strategy"}.issubset(cov_df.columns):
        csub = cov_df[(cov_df["symbol"] == symbol) & (cov_df["strategy"] == strategy)]
        if not csub.empty:
            clast = csub.tail(1).iloc[0]
            val_num_classes = _i(clast.get("num_classes"), 0) or 0
            val_covered     = _i(clast.get("covered"), 0) or 0
            val_coverage    = _f(clast.get("coverage"), 0.0) or 0.0

    # -----------------------
    # 5) health 판정 (그대로)
    # -----------------------
    nan_reasons = []
    status_str = str(trow.get("status", "") or "")
    if (val_acc is None or val_f1 is None) and status_str.lower() != "success":
        nan_reasons.append("학습 도중 오류가 발생했거나 데이터가 부족해 성능이 계산되지 않았어요.")
    if int(label_classes) <= 1 and int(label_total) > 0:
        nan_reasons.append("라벨이 한 종류라서 모델이 구분 학습을 할 수 없어요.")

    health_codes = []
    if status_str.lower() not in {"success", "ok"}:
        health_codes.append("STATUS_FAIL")
    if (val_f1 is None) or (val_f1 <= 0):
        health_codes.append("F1_ZERO")
    if int(label_classes) <= 1 and int(label_total) > 0:
        health_codes.append("LABEL_SINGLE_CLASS")

    health = "OK" if not health_codes else ";".join(health_codes)

    # -----------------------
    # 7) 저장 row
    # -----------------------
    summary_row = {
        "timestamp": now_kst().isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "model": model,

        "val_acc": 0.0 if val_acc is None else float(val_acc),
        "val_f1": 0.0 if val_f1 is None else float(val_f1),
        "val_loss": "" if val_loss is None else float(val_loss),

        "label_total": int(label_total),
        "label_classes": int(label_classes),
        "label_entropy": 0.0 if label_entropy is None else float(label_entropy),
        "label_counts_json": label_counts_json,

        "class_ranges_text": class_ranges_text,

        "near_zero_band": float(_f(trow.get("near_zero_band"), 0.0) or 0.0),
        "near_zero_count": int(_i(trow.get("near_zero_count"), 0) or 0),

        "data_rows": trow.get("rows", ""),
        "enough_for_training": trow.get("enough_for_training", ""),
        "augment_needed": trow.get("augment_needed", ""),

        # ✅ return_distribution 값: 없으면 빈값
        "ret_min": "" if ret_min is None else float(ret_min),
        "ret_p25": "" if ret_p25 is None else float(ret_p25),
        "ret_p50": "" if ret_p50 is None else float(ret_p50),
        "ret_p75": "" if ret_p75 is None else float(ret_p75),
        "ret_p90": "" if ret_p90 is None else float(ret_p90),
        "ret_p95": "" if ret_p95 is None else float(ret_p95),
        "ret_p99": "" if ret_p99 is None else float(ret_p99),
        "ret_max": "" if ret_max is None else float(ret_max),
        "ret_count": int(ret_count),

        "val_num_classes": int(val_num_classes),
        "val_covered": int(val_covered),
        "val_coverage": float(val_coverage),

        "status": status_str,
        "note": trow.get("note", ""),
        "health": health,
        "nan_reasons": " | ".join(nan_reasons),
    }

    # 8) 저장(기존 유지)
    df_old = _safe_read_df(out_path)
    if not df_old.empty:
        df_old = df_old[
            ~(
                (df_old["symbol"] == symbol) &
                (df_old["strategy"] == strategy) &
                (df_old["model"] == model)
            )
        ]

    df_new = pd.concat([df_old, pd.DataFrame([summary_row])], ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_new.to_csv(out_path, index=False, encoding="utf-8-sig")



# 🔥🔥🔥 여기부터 추가: /train-log 카드용 요약 함수 🔥🔥🔥
def get_train_log_cards(max_cards: int = 200):
    """
    /train-log 화면용 헬퍼.

    1순위: logs/train_dashboard.csv 사용
    2순위: 만약 이 파일이 없거나 비어 있으면,
           persistent/logs/train_log.csv 를 직접 읽어서
           심볼·전략별 '최근 한 줄'만 카드로 만든다.
    """
    dash_path = os.path.join(LOG_DIR, "train_dashboard.csv")
    raw_path = TRAIN_LOG

    # 1) 대시보드 우선
    df = _safe_read_df(dash_path)

    # 2) fallback: train_dashboard 가 비어 있으면 train_log.csv 직접 사용
    if df.empty:
        raw = _safe_read_df(raw_path)
        if raw.empty:
            return []

        raw = raw.copy()

        if "val_acc" not in raw.columns and "accuracy" in raw.columns:
            raw["val_acc"] = raw["accuracy"]
        if "val_f1" not in raw.columns and "f1" in raw.columns:
            raw["val_f1"] = raw["f1"]
        if "val_loss" not in raw.columns and "loss" in raw.columns:
            raw["val_loss"] = raw["loss"]

        # ✅ fallback도 깨진 model(all/trainer/none 등) 제외
        if "model" in raw.columns:
            bad = {"all", "trainer", "none", "nan", "null", ""}
            raw["_m"] = raw["model"].astype(str).str.strip().str.lower()
            raw = raw[~raw["_m"].isin(bad)].copy()

        if raw.empty:
            return []

        df = pd.DataFrame()
        df["timestamp"] = raw.get("timestamp", "")
        df["symbol"] = raw.get("symbol", "")
        df["strategy"] = raw.get("strategy", "")
        df["model"] = raw.get("model", "")

        df["val_acc"] = raw.get("val_acc", 0.0)
        df["val_f1"] = raw.get("val_f1", 0.0)
        df["val_loss"] = raw.get("val_loss", "")

        # label_total은 rows로 대충 채우되(표시용), classes는 모르면 0
        df["label_total"] = raw.get("rows", 0)
        df["label_classes"] = 0
        df["label_entropy"] = ""
        df["label_counts_json"] = ""

        df["enough_for_training"] = raw.get("enough_for_training", "")
        df["augment_needed"] = raw.get("augment_needed", "")

        # ✅ 핵심: fallback에서 수익률 요약값을 0.0으로 “거짓” 채우지 말고 빈값
        df["ret_min"] = ""
        df["ret_p25"] = ""
        df["ret_p50"] = ""
        df["ret_p75"] = ""
        df["ret_p90"] = ""
        df["ret_p95"] = ""
        df["ret_p99"] = ""
        df["ret_max"] = ""
        df["ret_count"] = ""

        df["val_num_classes"] = 0
        df["val_covered"] = 0
        df["val_coverage"] = 0.0

        df["class_ranges_text"] = ""

        df["near_zero_band"] = raw.get("near_zero_band", 0.0)
        df["near_zero_count"] = raw.get("near_zero_count", 0)

        # ✅ 핵심: status 기본값을 success로 “거짓” 주지 말고 unknown
        df["status"] = raw.get("status", "unknown")
        df["note"] = raw.get("note", "")

        if "health" in raw.columns:
            df["health"] = raw.get("health")
        else:
            df["health"] = df["status"].fillna("unknown")

        # rows 컬럼 그대로 들고 오기 (데이터 양 표시에 쓸 것)
        df["data_rows"] = raw.get("rows", "")

    if df.empty or "symbol" not in df.columns or "strategy" not in df.columns:
        return []

    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    cards = []

    for (sym, strat), g in df.groupby(["symbol", "strategy"], dropna=False):
        sym_str = str(sym or "").strip()
        strat_str = str(strat or "").strip()
        if sym_str.lower() in {"", "nan", "none", "null"} or strat_str.lower() in {"", "nan", "none", "null"}:
            continue

        if g.empty:
            continue
        last = g.iloc[-1]

        def _f(row, key, default=0.0):
            try:
                val = row.get(key, default)
                if val in ["", None, "nan", "NaN"]:
                    return float(default)
                return float(val)
            except Exception:
                return float(default)

        # ✅ FIX: "12.0" 같은 값도 안전하게 int로 변환
        def _i(row, key, default=0):
            try:
                val = row.get(key, default)
                if val in ["", None, "nan", "NaN"]:
                    return int(default)
                if isinstance(val, str):
                    s = val.strip()
                    if s == "" or s.lower() in {"nan", "none", "null"}:
                        return int(default)
                    return int(float(s))
                return int(float(val))
            except Exception:
                return int(default)

        val_acc = _f(last, "val_acc", 0.0)
        val_f1 = _f(last, "val_f1", 0.0)
        val_loss = last.get("val_loss", "")
        try:
            if val_loss in ["", None, "nan", "NaN"]:
                val_loss = 0.0
            val_loss = float(val_loss)
        except Exception:
            val_loss = 0.0

        # 🔹 라벨/클래스 관련 값들
        label_total = _i(last, "label_total", 0)
        label_classes = _i(last, "label_classes", 0)
        label_counts_json = str(last.get("label_counts_json", "") or "")

        # 🔹 검증 커버리지 관련
        val_num_classes = _i(last, "val_num_classes", 0)
        val_covered = _i(last, "val_covered", 0)
        val_coverage = _f(last, "val_coverage", 0.0)

        # 🔹 near-zero 수익률 구간
        near_zero_band = _f(last, "near_zero_band", 0.0)
        near_zero_count = _i(last, "near_zero_count", 0)

        # 🔹 원본 rows 표시용
        data_rows_raw = str(last.get("data_rows", last.get("rows", "")) or "").strip()
        if data_rows_raw.lower() in {"nan", "none", "null"}:
            data_rows_raw = ""

        health = str(last.get("health", "OK") or "OK")
        status = str(last.get("status", "") or "")

        enough_for_training = str(last.get("enough_for_training", "") or "")
        augment_needed = str(last.get("augment_needed", "") or "")

        # 1) 건강 상태 텍스트
        if health == "OK":
            health_text = "✅ 정상 학습: 데이터와 모델에 큰 문제 없이 학습이 잘 끝났어요."
        else:
            human_reasons = []
            if "STATUS_FAIL" in health:
                human_reasons.append("학습 과정에서 오류나 실패가 있었어요.")
            if "F1_ZERO" in health:
                human_reasons.append("모델이 정답 패턴을 거의 못 찾고 있어요(F1=0).")
            if "LABEL_SINGLE_CLASS" in health:
                human_reasons.append("라벨이 한 종류만 있어서 구분 학습이 불가능해요.")
            if "LOW_COVERAGE" in health:
                human_reasons.append("검증에서 일부 클래스만 등장해서 성능을 믿기 어려워요.")
            if not human_reasons:
                human_reasons.append("상세 원인은 health 코드에 들어 있어요.")

            health_text = "⚠️ 문제 있는 학습: " + " ".join(human_reasons)

        # 2) accuracy / F1 / loss 설명
        acc_text = f"정답률(accuracy): {val_acc*100:.1f}% — 전체 예측 중 정답으로 맞춘 비율이에요."
        f1_text = f"F1 점수: {val_f1*100:.1f}% — 정답률과 재현율을 합쳐서 '패턴을 제대로 배우고 있는지' 보는 지표예요."
        loss_text = f"손실(loss): {val_loss:.4f} — 낮을수록 좋고, 0에 가까울수록 모델이 더 안정적으로 학습된 거예요."

        # 3) 데이터 요약
        if label_total > 0 and label_classes > 0:
            data_summary = f"학습에 사용한 데이터: 총 {label_total}개, 구분한 수익률 구간(클래스): {label_classes}개."
        elif label_total > 0:
            data_summary = f"학습에 사용한 데이터: 총 {label_total}개 (클래스 개수 정보는 아직 없어요)."
        elif data_rows_raw not in ["", "0"]:
            data_summary = f"학습에 사용한 원본 샘플(캔들 기준): 약 {data_rows_raw}개."
        else:
            data_summary = "학습에 사용된 데이터 양 정보를 아직 찾지 못했어요."

        extra_data_info = []
        if enough_for_training not in ["", "0", "False", "false", "NO", "no"]:
            extra_data_info.append("✔ 이 정도 데이터로도 학습하기에 '충분'하다고 판단했어요.")
        else:
            extra_data_info.append("⚠ 데이터 양이 충분하지 않을 수 있어서, 결과를 신중하게 봐야 해요.")

        if augment_needed not in ["", "0", "False", "false", "NO", "no"]:
            extra_data_info.append("✔ 부족한 구간은 '데이터 증강'으로 채워줬어요.")
        else:
            extra_data_info.append("ℹ 이번 학습에서는 별도의 데이터 증강은 사용하지 않았어요.")

        data_detail_text = " ".join(extra_data_info)

        # 4) 수익률 요약 (빈값이면 “없음” 처리)
        ret_min = last.get("ret_min", "")
        ret_p50 = last.get("ret_p50", "")
        ret_max = last.get("ret_max", "")
        try:
            if ret_min in ["", None, "nan", "NaN"] or ret_p50 in ["", None, "nan", "NaN"] or ret_max in ["", None, "nan", "NaN"]:
                raise ValueError("empty")
            ret_min_f = float(ret_min); ret_p50_f = float(ret_p50); ret_max_f = float(ret_max)
            ret_summary_text = (
                f"수익률 분포: 최소 {ret_min_f*100:.2f}% ~ 최대 {ret_max_f*100:.2f}%, "
                f"중앙값은 {ret_p50_f*100:.2f}% 근처예요."
            )
        except Exception:
            ret_summary_text = "수익률 분포 정보는 아직 정리되지 않았어요."

        if near_zero_band > 0 and near_zero_count > 0:
            ret_summary_text += f" 0% ±{near_zero_band*100:.2f}% 구간에 데이터 {near_zero_count}개가 모여 있어요."

        # 5) 검증 커버리지 요약
        if val_num_classes > 0:
            coverage_summary = (
                f"검증 커버리지: 전체 {val_num_classes}개 구간 중 "
                f"{val_covered}개 구간이 실제 검증 데이터에 등장했어요 "
                f"({val_coverage*100:.1f}%)."
            )
        else:
            if label_classes > 0:
                coverage_summary = (
                    f"검증 커버리지는 따로 집계되지 않았지만, "
                    f"현재 라벨링된 수익률 구간은 총 {label_classes}개예요."
                )
            else:
                coverage_summary = "검증에서 각 수익률 구간이 얼마나 나왔는지는 아직 집계되지 않았어요."

        # 6) 클래스별 수익률 구간 텍스트
        class_ranges_text = str(last.get("class_ranges_text", "") or "")
        if label_classes <= 1:
            class_ranges_text_human = ""
        elif class_ranges_text:
            class_ranges_text_human = "각 클래스별 수익률 구간: " + class_ranges_text
        else:
            class_ranges_text_human = ""

        # 7) 초보용 요약
        beginner_summary = []
        if health == "OK":
            beginner_summary.append("👉 요약: 이 심볼/전략은 일단 '학습은 정상적으로 끝났고' 기본 성능도 무난한 편이에요.")
        else:
            beginner_summary.append("👉 요약: 이 심볼/전략은 학습 과정이나 데이터 쪽에 한 번 더 점검이 필요한 상태예요.")

        if val_acc >= 0.6 and val_f1 >= 0.4:
            beginner_summary.append("정답률과 패턴 인식(F1)도 어느 정도는 올라온 상태라, 이후 예측 결과를 보면서 튜닝하면 좋아요.")
        elif val_acc == 0.0 and val_f1 == 0.0:
            beginner_summary.append("정답률 / F1 이 거의 0이라, 라벨링이나 데이터 분포를 먼저 확인해보는 게 좋아요.")
        else:
            beginner_summary.append("성능이 애매한 구간이라, 데이터 양과 라벨 분포, 수익률 구간이 고르게 분포했는지 함께 보는 게 좋아요.")

        beginner_summary_text = " ".join(beginner_summary)

        card = {
            "symbol": sym_str,
            "strategy": strat_str,
            "model": str(last.get("model", "") or ""),

            "health": health,
            "health_text": health_text,
            "status": status,

            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_loss": val_loss,

            "label_total": label_total,
            "label_classes": label_classes,
            "label_counts_json": label_counts_json,
            "num_classes": label_classes,
            "class_count": label_classes,

            "data_summary": data_summary,
            "data_detail_text": data_detail_text,

            "enough_for_training": enough_for_training,
            "augment_needed": augment_needed,

            "val_num_classes": val_num_classes,
            "val_covered": val_covered,
            "val_coverage": val_coverage,
            "coverage_summary": coverage_summary,
            "all_classes_covered": bool(val_num_classes > 0 and val_covered >= val_num_classes),

            "class_ranges_text": class_ranges_text_human,
            "ret_summary_text": ret_summary_text,

            "near_zero_band": near_zero_band,
            "near_zero_count": near_zero_count,

            "timestamp": str(last.get("timestamp", "")),
            "note": str(last.get("note", "") or ""),

            "acc_text": acc_text,
            "f1_text": f1_text,
            "loss_text": loss_text,

            "beginner_summary_text": beginner_summary_text,
        }

        cards.append(card)

    cards = sorted(cards, key=lambda c: (c["symbol"], c["strategy"], c["timestamp"]))
    if max_cards is not None and len(cards) > max_cards:
        cards = cards[-max_cards:]

    return cards



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
        usecols = ["timestamp","symbol","strategy","model","status","success","source"]
        for chunk in pd.read_csv(
            PREDICTION_LOG, encoding="utf-8-sig",
            usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success","source"]],
            chunksize=CHUNK
        ):
            # source 필터: 훈련/디버그 소스 제외
            if "source" in chunk.columns:
                chunk = chunk[~chunk["source"].astype(str).isin(LOG_SOURCE_BLACKLIST)]
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
                    # 훈련/디버그 소스 제외
                    df = df[~df[src_col].astype(str).isin(LOG_SOURCE_BLACKLIST)]
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

# ============================================================
# 수익률 분포 추출/히스토그램 (labels.py 수식과 최대한 통일)
# ============================================================

# labels.py 의 helper 를 가져와서,
# 운영 로그 수익률 계산을 학습 라벨 분포와 맞춘다.
try:
    from labels import (
        _strategy_horizon_candles_from_hours as _lbl_strategy_horizon_candles_from_hours,
        _future_extreme_signed_returns_by_candles as _lbl_future_extreme_signed_returns_by_candles,
        _infer_bar_hours_from_df as _lbl_infer_bar_hours_from_df,
        _build_bins as _lbl_build_bins,
        _auto_target_bins as _lbl_auto_target_bins,
        compute_label_returns as _lbl_compute_label_returns,
    )
except Exception:
        _lbl_strategy_horizon_candles_from_hours = None
        _lbl_future_extreme_signed_returns_by_candles = None
        _lbl_infer_bar_hours_from_df = None
        _lbl_build_bins = None
        _lbl_auto_target_bins = None
        _lbl_compute_label_returns = None

def extract_candle_returns(
    df,
    max_rows: int = 1000,
    strategy: str | None = None,
    horizon_hours: int | None = None,
    symbol: str | None = None,
):
    """
    학습 labels.py 와 최대한 동일한 방식으로 '미래 구간 수익률'을 추출한다.

    - 기본 아이디어 (labels.py 와 동일):
      1) 전략(strategy)에 맞는 horizon 을 캔들 개수로 변환
      2) 해당 horizon 동안의 future high/low 수익률(up/dn)을 계산
      3) 분포용 값은 [dn, up] 을 모두 이어붙인 배열
         → labels.make_labels() 의 dist_for_bins 와 완전히 같은 개념

    - strategy / horizon_hours 를 안 넘기면:
      → 예전 방식(각 캔들의 high/low vs close)을 fallback 으로 사용.
    """
    if df is None or getattr(df, "empty", True):
        return []

    try:
        df_use = df.tail(max_rows).copy()
    except Exception:
        df_use = df

    # --- 1) labels.compute_label_returns 기반: 학습 라벨과 완전 동일한 수익률 정의 ---
    try:
        if _lbl_compute_label_returns is not None and strategy is not None:
            gains, up_c, dn_c, _dyn_bins = _lbl_compute_label_returns(
                df_use,
                symbol or "UNKNOWN",
                strategy,
            )
            dist = np.concatenate([dn_c, up_c], axis=0).astype(float)
            dist = dist[np.isfinite(dist)]
            return dist.tolist()
    except Exception as e:
        print(f"[logger.extract_candle_returns] compute_label_returns 기반 계산 실패 → labels helper fallback 사용 ({e})")

    # --- 2) labels helper 기반: 미래 구간(high/low) 수익률 (이전 코드 유지) ---
    try:
        if _lbl_future_extreme_signed_returns_by_candles is not None:
            horizon_candles = None

            # (a) strategy 기준 (단기/중기/장기)
            if strategy is not None and _lbl_strategy_horizon_candles_from_hours is not None:
                try:
                    horizon_candles = int(
                        max(1, _lbl_strategy_horizon_candles_from_hours(df_use, strategy))
                    )
                except Exception:
                    horizon_candles = None

            # (b) strategy 없고 horizon_hours 직접 넘어온 경우
            if horizon_candles is None and horizon_hours is not None:
                bar_h = 1.0
                if _lbl_infer_bar_hours_from_df is not None:
                    try:
                        bar_h = float(_lbl_infer_bar_hours_from_df(df_use))
                    except Exception:
                        bar_h = 1.0
                else:
                    # labels helper 를 못쓰는 경우, 단순 시간차로 추정
                    try:
                        ts = pd.to_datetime(df_use["timestamp"], errors="coerce").sort_values()
                        diffs = ts.diff().dropna()
                        bar_h = float(diffs.median().total_seconds() / 3600.0) if not diffs.empty else 1.0
                    except Exception:
                        bar_h = 1.0

                if not (bar_h > 0 and np.isfinite(bar_h)):
                    bar_h = 1.0
                horizon_candles = max(1, int(round(float(horizon_hours) / bar_h)))

            if horizon_candles is None:
                # 정보가 전혀 없으면 최소 1캔들
                horizon_candles = 1

            up_c, dn_c = _lbl_future_extreme_signed_returns_by_candles(df_use, int(horizon_candles))
            dist = np.concatenate([dn_c, up_c], axis=0).astype(float)
            dist = dist[np.isfinite(dist)]
            return dist.tolist()
    except Exception as e:
        print(f"[logger.extract_candle_returns] labels 기반 계산 실패 → fallback 사용 ({e})")

    # --- 3) 완전 fallback: 기존 per-candle high/low 방식 (과거 코드 유지) ---
    rets: list[float] = []
    for _, row in df_use.iterrows():
        try:
            base = float(row["close"])
            high_ = float(row.get("high", row["close"]))
            low_ = float(row.get("low", row["close"]))
        except Exception:
            continue

        if not np.isfinite(base) or base <= 0:
            continue

        up_ret = (high_ - base) / base   # 위로 간 수익률
        dn_ret = (low_ - base) / base    # 아래로 간 수익률

        if np.isfinite(up_ret):
            rets.append(float(up_ret))
        if np.isfinite(dn_ret):
            rets.append(float(dn_ret))

    return rets

def make_return_histogram(returns: list[float], bins: int = 20):
    """
    수익률 리스트를 받아서 히스토그램(구간, 개수)으로 바꿔준다.
    - labels.py 의 _build_bins / _auto_target_bins 를 사용할 수 있으면 최대한 재사용해서
      '학습 라벨 분포와 동일한 방식'으로 bin 경계를 만든다.
    - returns 가 비어있으면 빈 결과 반환.
    """
    if not returns:
        return {
            "bin_edges": [],
            "bin_counts": [],
        }

    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "bin_edges": [],
            "bin_counts": [],
        }

    # 1) labels.py 의 _build_bins / _auto_target_bins 를 사용할 수 있으면 그대로 사용
    if _lbl_build_bins is not None and _lbl_auto_target_bins is not None:
        try:
            # dist_for_bins 는 [dn, up] 을 이어붙인 배열이므로,
            # labels 쪽에서는 "샘플 수 N" 을 기준으로 bin 개수를 정한다.
            # 여기서는 dist 길이(arr.size)를 그대로 넘기되,
            # 최소 2개 이상 bin 을 보장한다.
            approx_n = max(1, int(arr.size))
            dynamic_bins = int(_lbl_auto_target_bins(approx_n))
            dynamic_bins = max(2, int(dynamic_bins))

            edges, counts, _spans = _lbl_build_bins(arr, dynamic_bins)
            return {
                "bin_edges": edges.astype(float).tolist(),
                "bin_counts": counts.astype(int).tolist(),
            }
        except Exception as e:
            print(f"[logger.make_return_histogram] labels._build_bins 사용 실패 → fallback 사용 ({e})")

    # 2) fallback: 기존의 단순 np.histogram 방식 (호환용)
    try:
        counts, edges = np.histogram(arr, bins=int(max(2, bins)))
    except Exception:
        counts, edges = np.histogram(arr, bins=20)

    return {
        "bin_edges": edges.astype(float).tolist(),
        "bin_counts": counts.astype(int).tolist(),
                }
