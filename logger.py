# === logger.py (메모리 안전: 로테이션 + 청크 집계 + 모델명 정규화 + 실패DB 노이즈 차단 + 파일락 + 컨텍스트 분기 최종본) ===
import os, csv, json, datetime, pandas as pd, pytz, hashlib, shutil
import sqlite3
from collections import defaultdict
import threading, time  # 동시성/재시도

# -------------------------
# 기본 경로/디렉토리
# -------------------------
DIR = "/persistent"
LOG_DIR = os.path.join(DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# prediction_log는 루트 경로로 통일
PREDICTION_LOG = f"{DIR}/prediction_log.csv"
WRONG = f"{DIR}/wrong_predictions.csv"
EVAL_RESULT = f"{LOG_DIR}/evaluation_result.csv"

# 학습 로그 파일명
TRAIN_LOG = f"{LOG_DIR}/train_log.csv"
AUDIT_LOG = f"{LOG_DIR}/evaluation_audit.csv"

# 공용 헤더(기존 + 확장 컬럼)
BASE_PRED_HEADERS = [
    "timestamp","symbol","strategy","direction",
    "entry_price","target_price",
    "model","predicted_class","top_k","note",
    "success","reason","rate","return_value",
    "label","group_id","model_symbol","model_name",
    "source","volatility","source_exchange"
]
EXTRA_PRED_HEADERS = ["regime","meta_choice","raw_prob","calib_prob","calib_ver"]
# ✅ feature_vector + (NEW) class return 3컬럼
CLASS_RANGE_HEADERS = ["class_return_min","class_return_max","class_return_text"]
PREDICTION_HEADERS = BASE_PRED_HEADERS + EXTRA_PRED_HEADERS + ["feature_vector"] + CLASS_RANGE_HEADERS

# 청크 크기 기본값
CHUNK = 50_000

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# -------------------------
# (NEW) 간단 파일락 유틸 (멀티프로세스 동시쓰기 보호)
# -------------------------
_PRED_LOCK_PATH = PREDICTION_LOG + ".lock"
_LOCK_STALE_SEC = 120  # 고아 락 정리 기준

class _FileLock:
    def __init__(self, path: str, timeout: float = 10.0, poll: float = 0.05):
        self.path = path
        self.timeout = float(timeout)
        self.poll = float(poll)

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                # 존재하는 고아 락(너무 오래된 락)은 제거
                if os.path.exists(self.path):
                    try:
                        mtime = os.path.getmtime(self.path)
                        if (time.time() - mtime) > _LOCK_STALE_SEC:
                            os.remove(self.path)
                    except Exception:
                        pass
                # 원자적 생성 시도
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} ts={time.time()}\n")
                break  # 획득 성공
            except FileExistsError:
                if time.time() >= deadline:
                    # 마지막 시도: 고아락 판단되면 제거하고 재시도
                    try:
                        mtime = os.path.getmtime(self.path)
                        if (time.time() - mtime) > _LOCK_STALE_SEC:
                            os.remove(self.path)
                            continue
                    except Exception:
                        pass
                    raise TimeoutError(f"lock timeout: {self.path}")
                time.sleep(self.poll)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass

# -------------------------
# 안전한 로그 파일 보장
# -------------------------
def _read_csv_header(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first = f.readline().strip()
        if not first:
            return []
        return [h.strip() for h in first.split(",")]
    except Exception:
        return []

def ensure_prediction_log_exists():
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
                os.replace(PREDICTION_LOG, bak)
                with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as out, \
                     open(bak, "r", encoding="utf-8-sig") as src:
                    w = csv.writer(out); w.writerow(PREDICTION_HEADERS)
                    reader = csv.reader(src)
                    try: next(reader)  # old header skip
                    except StopIteration: reader = []
                    for row in reader:
                        row = (row + [""] * len(PREDICTION_HEADERS))[:len(PREDICTION_HEADERS)]
                        w.writerow(row)
                print("[✅ ensure_prediction_log_exists] 기존 파일 헤더 보정(확장) 완료")
    except Exception as e:
        print(f"[⚠️ ensure_prediction_log_exists] 예외: {e}")

# -------------------------
# (NEW) 용량 기반 로그 로테이션
# -------------------------
def rotate_prediction_log_if_needed(max_mb: int = 200, backups: int = 3):
    try:
        if not os.path.exists(PREDICTION_LOG):
            return
        size_mb = os.path.getsize(PREDICTION_LOG) / (1024 * 1024)
        if size_mb < max_mb:
            return
        for i in range(backups, 0, -1):
            old = f"{PREDICTION_LOG}.{i}"
            if i == backups and os.path.exists(old):
                os.remove(old)
            else:
                prev = f"{PREDICTION_LOG}.{i-1}" if i-1 > 0 else PREDICTION_LOG
                if os.path.exists(prev):
                    shutil.move(prev, old)
        ensure_prediction_log_exists()
        print(f"[logger] 🔁 rotate: {size_mb:.1f}MB → rotated with {backups} backups")
    except Exception as e:
        print(f"[logger] rotate error: {e}")

# -------------------------
# feature hash 유틸
# -------------------------
def get_feature_hash(feature_row) -> str:
    try:
        import numpy as _np
        if feature_row is None:
            return "none"
        if "torch" in str(type(feature_row)):
            try:
                feature_row = feature_row.detach().cpu().numpy()
            except Exception:
                pass
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
# SQLite: 모델 성공/실패 집계
# -------------------------
_db_conn = None
_DB_PATH = os.path.join(LOG_DIR, "failure_patterns.db")
_db_lock = threading.RLock()

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
    conn = sqlite3.connect(_DB_PATH, timeout=30, check_same_thread=False)
    _apply_sqlite_pragmas(conn)
    return conn

def get_db_connection():
    global _db_conn
    with _db_lock:
        if _db_conn is None:
            try:
                _db_conn = _connect_sqlite()
                print("[✅ logger.py DB connection 생성 완료]")
            except Exception as e:
                print(f"[오류] logger.py DB connection 생성 실패 → {e}")
                _db_conn = None
        return _db_conn

def _sqlite_exec_with_retry(sql, params=(), retries=5, sleep_base=0.2, commit=False):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with _db_lock:
                conn = get_db_connection()
                if conn is None:
                    conn = _connect_sqlite()
                    globals()['_db_conn'] = conn
                cur = conn.cursor()
                cur.execute(sql, params)
                if commit:
                    conn.commit()
                try:
                    rows = cur.fetchall()
                except sqlite3.ProgrammingError:
                    rows = None
                cur.close()
                return rows
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            last_err = e
            if ("database is locked" in msg) or ("disk i/o error" in msg) or ("database is busy" in msg):
                with _db_lock:
                    try:
                        if globals().get("_db_conn"):
                            try:
                                globals()["_db_conn"].close()
                            except Exception:
                                pass
                        globals()["_db_conn"] = _connect_sqlite()
                    except Exception as ce:
                        last_err = ce
                time.sleep(sleep_base * attempt)
                continue
            else:
                raise
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * attempt)
            continue
    raise last_err if last_err else RuntimeError("sqlite exec failed")

def ensure_success_db():
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

def update_model_success(s, t, m, success):
    try:
        _sqlite_exec_with_retry("""
            INSERT INTO model_success (symbol, strategy, model, success, fail)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, strategy, model) DO UPDATE SET
                success = success + excluded.success,
                fail = fail + excluded.fail
        """, params=(s, t or "알수없음", m, int(success), int(not success)), retries=7, commit=True)
        print(f"[✅ update_model_success] {s}-{t}-{m} 기록 ({'성공' if success else '실패'})")
    except Exception as e:
        print(f"[오류] update_model_success 실패 → {e}")

def get_model_success_rate(s, t, m):
    try:
        rows = _sqlite_exec_with_retry("""
            SELECT success, fail FROM model_success
            WHERE symbol=? AND strategy=? AND model=?
        """, params=(s, t or "알수없음", m), retries=5, commit=False)
        row = rows[0] if rows else None
        if row is None:
            return 0.0
        success_cnt, fail_cnt = row
        total = success_cnt + fail_cnt
        return (success_cnt / total) if total > 0 else 0.0
    except Exception as e:
        print(f"[오류] get_model_success_rate 실패 → {e}")
        return 0.0

# -------------------------
# failure_db 초기화: 부팅 직후 반드시 스키마 준비(명시 로그)
# -------------------------
try:
    from failure_db import ensure_failure_db as _ensure_failure_db_once
    _ensure_failure_db_once()
    print("[logger] failure_db initialized (schema ready)")
except Exception as _e:
    print(f"[logger] failure_db init failed: {_e}")

# 서버 시작 시 보장
ensure_success_db()
ensure_prediction_log_exists()

# -------------------------
# 파일 로드/유틸
# -------------------------
def load_failure_count():
    path = os.path.join(LOG_DIR, "failure_count.csv")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}
    except:
        return {}

def _normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower().map(lambda x: "success" if x == "success" else "fail")
        return df
    if "success" in df.columns:
        s = df["success"].map(lambda x: str(x).strip().lower() in ["true","1","yes","y"])
        df["status"] = s.map(lambda b: "success" if b else "fail")
        return df
    df["status"] = ""
    return df

# -------------------------
# 메모리 안전 집계 (청크 기반)
# -------------------------
def get_meta_success_rate(strategy, min_samples: int = 1):
    if not os.path.exists(PREDICTION_LOG):
        return 0.0
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
        if chunk.empty:
            continue
        if "status" in chunk.columns and chunk["status"].notna().any():
            mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
            chunk = chunk[mask]
            s = chunk["status"].astype(str).str.lower().isin(["success","v_success"])
            succ += int(s.sum()); total += int(len(chunk))
        elif "success" in chunk.columns:
            s = chunk["success"].astype(str).str.lower().isin(["true","1","success","v_success"])
            succ += int(s.sum()); total += int(len(chunk))
    if total < max(1, min_samples):
        return 0.0
    return float(succ / total)

def get_strategy_eval_count(strategy: str):
    if not os.path.exists(PREDICTION_LOG):
        return 0
    usecols = ["strategy","status","success"]
    count = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success"]],
        chunksize=CHUNK
    ):
        if "strategy" in chunk.columns:
            chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty:
            continue
        if "status" in chunk.columns and chunk["status"].notna().any():
            mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
            count += int(mask.sum())
        elif "success" in chunk.columns:
            count += int(len(chunk))
    return int(count)

def get_actual_success_rate(strategy, min_samples: int = 1):
    if not os.path.exists(PREDICTION_LOG):
        return 0.0
    usecols = ["strategy","status","success"]
    succ = total = 0
    for chunk in pd.read_csv(
        PREDICTION_LOG, encoding="utf-8-sig",
        usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success"]],
        chunksize=CHUNK
    ):
        if "strategy" in chunk.columns:
            chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty:
            continue
        if "status" in chunk.columns and chunk["status"].notna().any():
            mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
            s = chunk["status"].astype(str).str.lower().isin(["success","v_success"])
            succ += int(s[mask].sum()); total += int(mask.sum())
        elif "success" in chunk.columns:
            s = chunk["success"].astype(str).str.lower().isin(["true","1","success","v_success"])
            succ += int(s.sum()); total += int(len(chunk))
    if total < max(1, min_samples):
        return 0.0
    return round(succ / total, 4)

# -------------------------
# 감사용 감사 로그
# -------------------------
def log_audit_prediction(s, t, status, reason):
    row = {
        "timestamp": now_kst().isoformat(),
        "symbol": str(s or "UNKNOWN"),
        "strategy": str(t or "알수없음"),
        "status": str(status),
        "reason": str(reason)
    }
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
    except:
        pass

# -------------------------
# 예측 로그 (확장 인자 지원 + 자동 컬럼맞춤 + 파일락)
# -------------------------
def _align_row_to_header(row, header):
    if len(row) < len(header):
        row = row + [""] * (len(header) - len(row))
    elif len(row) > len(header):
        row = row[:len(header)]
    return row

def _clean_str(x):
    s = str(x).strip() if x is not None else ""
    if s.lower() in {"", "unknown", "none", "nan", "null"}:
        return ""
    return s

def _normalize_model_fields(model, model_name, symbol, strategy):
    m = _clean_str(model)
    mn = _clean_str(model_name)
    if not m and mn:
        m = mn
    if not mn and m:
        mn = m
    if not m and not mn:
        base = f"auto_{symbol}_{strategy}"
        m = mn = base
    return m, mn

def log_prediction(
    symbol, strategy, direction=None, entry_price=0, target_price=0,
    timestamp=None, model=None, predicted_class=None, top_k=None,
    note="", success=False, reason="", rate=None, return_value=None,
    label=None, group_id=None, model_symbol=None, model_name=None,
    source="일반", volatility=False, feature_vector=None,
    source_exchange="BYBIT",
    # 확장 필드
    regime=None, meta_choice=None, raw_prob=None, calib_prob=None, calib_ver=None,
    # (NEW) 클래스 수익률 범위 3컬럼
    class_return_min=None, class_return_max=None, class_return_text=None,
):
    from datetime import datetime as _dt
    try:
        from failure_db import insert_failure_record
    except Exception:
        insert_failure_record = None

    LOG_FILE = PREDICTION_LOG
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with _FileLock(_PRED_LOCK_PATH, timeout=10.0):
        rotate_prediction_log_if_needed()
        ensure_prediction_log_exists()

        now = _dt.now(pytz.timezone("Asia/Seoul")).isoformat() if timestamp is None else timestamp
        top_k_str = ",".join(map(str, top_k)) if top_k else ""

        predicted_class = predicted_class if predicted_class is not None else -1
        label = label if label is not None else -1
        reason = (reason or "사유없음").strip()
        rate = 0.0 if rate is None else float(rate)
        return_value = 0.0 if return_value is None else float(return_value)
        entry_price = float(entry_price or 0.0)
        target_price = float(target_price or 0.0)
        model, model_name = _normalize_model_fields(model, model_name, symbol, strategy)

        allowed_sources = ["일반","기본","meta","evo_meta","baseline_meta","진화형","평가","단일","변동성","train_loop","섀도우"]
        if source not in allowed_sources:
            source = "일반"

        dir_s = str(direction or "")
        src_s = str(source or "")
        ctx = "evaluation" if (src_s == "평가" or dir_s.startswith("평가")) else "prediction"

        fv_serial = ""
        try:
            if feature_vector is not None:
                if isinstance(feature_vector, (list, tuple)):
                    v = list(feature_vector)
                elif "numpy" in str(type(feature_vector)):
                    v = feature_vector.flatten().tolist()
                else:
                    v = []
                if len(v) > 64:
                    fv_serial = json.dumps({"head": v[:8], "tail": v[-8:]}, ensure_ascii=False)
                else:
                    fv_serial = json.dumps(v, ensure_ascii=False)
        except Exception:
            fv_serial = ""

        row = [
            now, symbol, strategy, direction, entry_price, target_price,
            (model or ""), predicted_class, top_k_str, note,
            str(success), reason, rate, return_value, label,
            group_id, model_symbol, model_name, source, volatility, source_exchange,
            (regime or ""), (meta_choice or ""),
            (float(raw_prob) if raw_prob is not None else ""),
            (float(calib_prob) if calib_prob is not None else ""),
            (str(calib_ver) if calib_ver is not None else ""),
            fv_serial,
            # 새 컬럼 3개
            (float(class_return_min) if class_return_min is not None else ""),
            (float(class_return_max) if class_return_max is not None else ""),
            (str(class_return_text) if class_return_text is not None else ""),
        ]

        try:
            write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
            current_header = PREDICTION_HEADERS
            if not write_header:
                h = _read_csv_header(LOG_FILE)
                current_header = h if h else PREDICTION_HEADERS

            with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(PREDICTION_HEADERS)
                    current_header = PREDICTION_HEADERS
                w.writerow(_align_row_to_header(row, current_header))

            print(f"[✅ 예측 로그 기록됨] {symbol}-{strategy} class={predicted_class} | success={success} | src={source_exchange} | reason={reason}")

            should_record_failure = (
                insert_failure_record is not None
                and (ctx == "evaluation")
                and (not success)
                and (label not in (-1, "-1", None))
                and (entry_price not in (None, 0.0))
            )

            if should_record_failure:
                feature_hash = f"{symbol}-{strategy}-{model}-{predicted_class}-{label}-{rate}"
                safe_vector = []
                try:
                    import numpy as _np
                    if feature_vector is not None:
                        if isinstance(feature_vector, _np.ndarray):
                            safe_vector = feature_vector.flatten().tolist()
                        elif isinstance(feature_vector, list):
                            safe_vector = feature_vector
                except Exception:
                    safe_vector = []

                insert_failure_record(
                    {
                        "symbol": symbol, "strategy": strategy, "direction": direction,
                        "model": model, "predicted_class": predicted_class,
                        "rate": rate, "reason": reason, "label": label, "source": source,
                        "entry_price": entry_price, "target_price": target_price,
                        "return_value": return_value
                    },
                    feature_hash=feature_hash, label=label, feature_vector=safe_vector, context=ctx
                )
            else:
                if (insert_failure_record is None) or (ctx != "evaluation"):
                    log_audit_prediction(symbol, strategy, "skip_failure_db", f"ctx={ctx}, label={label}, entry_price={entry_price}")
        except Exception as e:
            print(f"[⚠️ 예측 로그 기록 실패] {e}")

# -------------------------
# 학습 로그
# -------------------------
def log_training_result(
    symbol, strategy, model="", accuracy=0.0, f1=0.0, loss=0.0,
    note="", source_exchange="BYBIT", status="success",
):
    LOG_FILE = TRAIN_LOG
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).isoformat()
    row = [
        now, str(symbol), str(strategy), str(model or ""),
        float(accuracy) if accuracy is not None else 0.0,
        float(f1) if f1 is not None else 0.0,
        float(loss) if loss is not None else 0.0,
        str(note or ""), str(source_exchange or "BYBIT"),
        str(status or "success")
    ]
    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","model","accuracy","f1","loss","note","source_exchange","status"])
            w.writerow(row)
        print(f"[✅ 학습 로그 기록] {symbol}-{strategy} {model} status={status}")
    except Exception as e:
        print(f"[⚠️ 학습 로그 기록 실패] {e}")

# -------------------------
# 수익률 클래스 경계 로그
# -------------------------
def log_class_ranges(symbol, strategy, group_id=None, class_ranges=None, note=""):
    import csv, os, math
    path = os.path.join(LOG_DIR, "class_ranges.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = now_kst().isoformat()

    class_ranges = class_ranges or []
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","idx","low","high","note"])
            for i, rng in enumerate(class_ranges):
                try:
                    lo, hi = (float(rng[0]), float(rng[1]))
                except Exception:
                    lo, hi = (None, None)
                w.writerow([now, symbol, strategy, int(group_id) if group_id is not None else 0, i, lo, hi, str(note or "")])
        print(f"[📐 클래스경계 로그] {symbol}-{strategy}-g{group_id} → {len(class_ranges)}개 기록")
    except Exception as e:
        print(f"[⚠️ 클래스경계 로그 실패] {e}")

# -------------------------
# 수익률 분포 요약 로그
# -------------------------
def log_return_distribution(symbol, strategy, group_id=None, horizon_hours=None, summary: dict=None, note=""):
    path = os.path.join(LOG_DIR, "return_distribution.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
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

    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","horizon_hours",
                            "min","p25","p50","p75","p90","p95","p99","max","count","note"])
            w.writerow(row)
        print(f"[📈 수익률분포 로그] {symbol}-{strategy}-g{group_id} count={s.get('count',0)}")
    except Exception as e:
        print(f"[⚠️ 수익률분포 로그 실패] {e}")

# -------------------------
# 라벨 분포 로그
# -------------------------
def log_label_distribution(
    symbol, strategy, group_id=None,
    counts: dict=None, total: int=None, n_unique: int=None, entropy: float=None,
    labels=None, note=""
):
    import json, math

    path = os.path.join(LOG_DIR, "label_distribution.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = now_kst().isoformat()

    if counts is None:
        from collections import Counter
        try:
            labels_list = list(map(int, list(labels or [])))
        except Exception:
            labels_list = []
        cnt = Counter(labels_list)
        total_calc = sum(cnt.values())
        probs = [c/total_calc for c in cnt.values()] if total_calc > 0 else []
        entropy_calc = -sum(p*math.log(p + 1e-12) for p in probs) if probs else 0.0
        counts = {int(k): int(v) for k, v in sorted(cnt.items())}
        total = total_calc
        n_unique = len(cnt)
        entropy = round(float(entropy_calc), 6)
    else:
        counts = {int(k): int(v) for k, v in sorted(counts.items())}
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
        int(total),
        json.dumps(counts, ensure_ascii=False),
        int(n_unique),
        float(entropy),
        str(note or "")
    ]

    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","symbol","strategy","group_id","total","counts_json","n_unique","entropy","note"])
            w.writerow(row)
        print(f"[📊 라벨분포 로그] {symbol}-{strategy}-g{group_id} → total={total}, classes={n_unique}, H={entropy:.4f}")
    except Exception as e:
        print(f"[⚠️ 라벨분포 로그 실패] {e}")

# -------------------------
# 모델 인벤토리 조회
# -------------------------
def get_available_models(symbol: str = None, strategy: str = None):
    try:
        model_dir = "/persistent/models"
        if not os.path.isdir(model_dir):
            return []
        out = []
        for fn in os.listdir(model_dir):
            if not fn.endswith(".pt"):
                continue
            pt_path = os.path.join(model_dir, fn)
            meta_path = pt_path.replace(".pt", ".meta.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            sym = meta.get("symbol") or fn.split("_", 1)[0]
            strat = meta.get("strategy") or ("단기" if "_단기_" in fn else "중기" if "_중기_" in fn else "장기" if "_장기_" in fn else "")
            if symbol and sym != symbol:
                continue
            if strategy and strat != strategy:
                continue
            out.append({
                "pt_file": fn,
                "meta_file": os.path.basename(meta_path),
                "symbol": sym,
                "strategy": strat,
                "model": meta.get("model", ""),
                "group_id": meta.get("group_id", 0),
                "num_classes": meta.get("num_classes", 0),
                "val_f1": float(meta.get("metrics", {}).get("val_f1", 0.0)),
                "timestamp": meta.get("timestamp", "")
            })
        out.sort(key=lambda r: (r["symbol"], r["strategy"], r["model"], r["group_id"]))
        return out
    except Exception as e:
        print(f"[오류] get_available_models 실패 → {e}")
        return []

# -------------------------
# 최근 예측 통계 내보내기 (청크 누산)
# -------------------------
def export_recent_model_stats(days: int = 7, out_path: str = None):
    try:
        ensure_prediction_log_exists()
        path = PREDICTION_LOG
        if out_path is None:
            os.makedirs(LOG_DIR, exist_ok=True)
            out_path = os.path.join(LOG_DIR, "recent_model_stats.csv")

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            pd.DataFrame(columns=["symbol","strategy","model","total","success","fail","success_rate","last_ts"]).to_csv(out_path, index=False, encoding="utf-8-sig")
            return out_path

        cutoff = now_kst() - datetime.timedelta(days=int(days))

        agg = {}

        usecols = ["timestamp","symbol","strategy","model","status","success"]
        for chunk in pd.read_csv(
            path, encoding="utf-8-sig",
            usecols=[c for c in usecols if c in PREDICTION_HEADERS or c in ["status","success"]],
            chunksize=CHUNK
        ):
            if "timestamp" in chunk.columns:
                ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
                try:
                    ts = ts.dt.tz_localize("Asia/Seoul")
                except Exception:
                    ts = ts.dt.tz_convert("Asia/Seoul")
                chunk = chunk.loc[ts >= cutoff]
                chunk = chunk.assign(_ts=ts)
            else:
                continue

            if chunk.empty or "model" not in chunk.columns:
                continue

            succ_mask = None
            if "status" in chunk.columns and chunk["status"].notna().any():
                ok_mask = chunk["status"].astype(str).str.lower().isin(["success","fail","v_success","v_fail"])
                succ_mask = chunk["status"].astype(str).str.lower().isin(["success","v_success"])
                chunk = chunk[ok_mask]
            elif "success" in chunk.columns:
                succ_mask = chunk["success"].astype(str).str.lower().isin(["true","1","success","v_success"])
            else:
                continue

            if chunk.empty:
                continue

            chunk = chunk.assign(_succ=succ_mask.astype(bool))

            for (sym, strat, mdl), sub in chunk.groupby(["symbol","strategy","model"], dropna=False):
                key = (str(sym), str(strat), str(mdl))
                d = agg.setdefault(key, {"success":0, "total":0, "last_ts":None})
                d["success"] += int(sub["_succ"].sum())
                d["total"]   += int(len(sub))
                last_ts = pd.to_datetime(sub["_ts"].max(), errors="coerce")
                if d["last_ts"] is None or (pd.notna(last_ts) and last_ts > d["last_ts"]):
                    d["last_ts"] = last_ts

        rows = []
        for (sym,strat,mdl), d in agg.items():
            total = int(d["total"]); succ = int(d["success"])
            rate = (succ/total) if total>0 else 0.0
            last_ts = d["last_ts"].isoformat() if d["last_ts"] is not None else ""
            rows.append({
                "symbol": sym, "strategy": strat, "model": mdl,
                "total": total, "success": succ, "fail": total - succ,
                "success_rate": round(rate, 4), "last_ts": last_ts
            })

        df_out = pd.DataFrame(rows, columns=["symbol","strategy","model","total","success","fail","success_rate","last_ts"])
        df_out = df_out.sort_values(["symbol","strategy","model","last_ts"])
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[✅ export_recent_model_stats] 저장: {out_path} (rows={len(df_out)})")
        return out_path
    except Exception as e:
        print(f"[⚠️ export_recent_model_stats 실패] {e}")
        try:
            pd.DataFrame(columns=["symbol","strategy","model","total","success","fail","success_rate","last_ts"]).to_csv(
                out_path or os.path.join(LOG_DIR, "recent_model_stats.csv"), index=False, encoding="utf-8-sig")
        except Exception:
            pass
        return out_path or os.path.join(LOG_DIR, "recent_model_stats.csv")
