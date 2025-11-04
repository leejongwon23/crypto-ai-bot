# === failure_db.py (v2025-10-26, 환경경로 반영판) =============================
# 실패 레코드 표준화 + CSV/SQLite 동시 기록 + 중복/폭주 방지 + 가벼운 분류태깅
#
# ✅ 핵심
# - wrong_predictions.csv 최소 스키마 보장(로더 호환)
# - SQLite 요약 테이블(failures) 동시 기록 / 중복키로 재기록 방지
# - 중복가드: feature_hash + (±90분, symbol, strategy, predicted_class)
# - 폭주가드(샘플링): 전략·시간창별 상한(단기/중기/장기 다르게) 초과 시 일부 드랍
# - 실패유형 태깅: negative_label, nan_label, prob_nan, class_out_of_range, bounds_mismatch, recur/evo/noise
# - 최근 패턴 유사도(코사인)로 recur/evo 라벨 부여(간단·경량)
# - 경보: 심각 사유는 콘솔 + alerts.log 기록
#
# 🔧 환경변수(선택):
#   FAIL_WIN_MINUTES=360            # 샘플링 윈도우(분)
#   FAIL_CAP_SHORT=40               # 단기(윈도우 내 최대 기록 수/심볼-전략)
#   FAIL_CAP_MID=20                 # 중기
#   FAIL_CAP_LONG=10                # 장기
#   FAIL_SIM_TOPK=200               # 유사도 계산 시 참조 상한
#   FAIL_SIM_RECUR=0.92             # recur 판정 임계치
#   FAIL_SIM_EVO=0.75               # evo 하한(이상은 evo, 그 미만은 noise 후보)
#   FAIL_NOISE_MIN_RET=0.001        # noise 판정용 |return_value| 하한
#
# 외부에서 사용하는 공개 함수(기존 호환):
#   ensure_failure_db()
#   check_failure_exists(row: dict) -> bool
#   load_existing_failure_hashes() -> set
#   insert_failure_record(record, feature_hash=None, label=None, feature_vector=None, context=None) -> bool
# ============================================================================

from __future__ import annotations
import os, csv, json, math, hashlib, time, threading, datetime, sqlite3
from typing import Any, Dict, Optional, Iterable, Tuple, List

import pandas as pd
import numpy as np

try:
    import pytz
except Exception:
    pytz = None

# ----------------------------------------------------------
# config 기반 경로 가져오기 (없으면 기존 /persistent로 폴백)
# ----------------------------------------------------------
_DEFAULT_DIR = "/persistent"
_DEFAULT_LOG_DIR = os.path.join(_DEFAULT_DIR, "logs")

try:
    # 이 함수들이 config에 없을 수도 있으니 개별 try/except로 감싼다
    from config import (
        get_FAILURE_DB_PATH,
        get_WRONG_PREDICTIONS_PATH,
        get_ALERT_LOG_PATH,
    )
except Exception:
    # 가져오지 못해도 아래에서 폴백하니까 그냥 넘어감
    get_FAILURE_DB_PATH = None
    get_WRONG_PREDICTIONS_PATH = None
    get_ALERT_LOG_PATH = None

# wrong_predictions.csv (CSV 원본)
if callable(get_WRONG_PREDICTIONS_PATH):
    WRONG_CSV = get_WRONG_PREDICTIONS_PATH()
else:
    WRONG_CSV = os.getenv("WRONG_PREDICTIONS_PATH", os.path.join(_DEFAULT_DIR, "wrong_predictions.csv"))

# SQLite DB 경로
if callable(get_FAILURE_DB_PATH):
    DB_PATH = get_FAILURE_DB_PATH()
else:
    DB_PATH = os.getenv("FAILURE_DB_PATH", os.path.join(_DEFAULT_LOG_DIR, "failure_records.db"))

# LOG_DIR 는 DB가 있는 디렉토리로 맞춰두면 안전
LOG_DIR = os.path.dirname(DB_PATH) if DB_PATH else _DEFAULT_LOG_DIR

# alerts.log
if callable(get_ALERT_LOG_PATH):
    ALERT_LOG = get_ALERT_LOG_PATH()
else:
    ALERT_LOG = os.getenv("ALERT_LOG_PATH", os.path.join(LOG_DIR, "alerts.log"))

# 디렉토리 보장
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(WRONG_CSV), exist_ok=True)

# ----------------------------------------------------------
# CSV 표준 헤더
# ----------------------------------------------------------
WRONG_HEADERS = [
    "timestamp","symbol","strategy","predicted_class","label",
    "model","group_id","entry_price","target_price","return_value",
    "reason","context","note","regime","meta_choice",
    "raw_prob","calib_prob","calib_ver",
    "feature_hash","feature_vector","source","source_exchange",
    # 확장 필드(있어도 무방):
    "failure_level","train_weight"
]

# 샘플링/유사도/노이즈 파라미터(환경변수 지원)
FAIL_WIN_MINUTES = int(os.getenv("FAIL_WIN_MINUTES", "360"))
FAIL_CAP_SHORT   = int(os.getenv("FAIL_CAP_SHORT", "40"))
FAIL_CAP_MID     = int(os.getenv("FAIL_CAP_MID", "20"))
FAIL_CAP_LONG    = int(os.getenv("FAIL_CAP_LONG", "10"))

FAIL_SIM_TOPK    = int(os.getenv("FAIL_SIM_TOPK", "200"))
FAIL_SIM_RECUR   = float(os.getenv("FAIL_SIM_RECUR", "0.92"))
FAIL_SIM_EVO     = float(os.getenv("FAIL_SIM_EVO", "0.75"))
FAIL_NOISE_MIN_RET = float(os.getenv("FAIL_NOISE_MIN_RET", "0.001"))

# 학습가중치(전략별/유형별 기본값)
BASE_WEIGHT = {
    "단기": {"recur": 0.8, "evo": 1.0, "noise": 0.0},
    "중기": {"recur": 0.6, "evo": 1.0, "noise": 0.0},
    "장기": {"recur": 0.4, "evo": 1.0, "noise": 0.0},
}

# ------------------------------ 시간 유틸 ------------------------------
def _now_kst() -> datetime.datetime:
    tz = pytz.timezone("Asia/Seoul") if pytz else None
    return datetime.datetime.now(tz) if tz else datetime.datetime.now()

def _now_kst_iso() -> str:
    return _now_kst().isoformat()

# ------------------------------ 해시/안전 변환 ------------------------------
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
    # 문자열 JSON이면 파싱 시도
    if isinstance(fv, str) and fv.strip().startswith("["):
        try:
            arr = np.array(json.loads(fv), dtype=float).reshape(-1)
            return _sha1_of_list(arr)
        except Exception:
            pass
    return "none"

# ------------------------------ 파일/DB 보장 ------------------------------
def _ensure_wrong_csv():
    os.makedirs(os.path.dirname(WRONG_CSV), exist_ok=True)
    if not os.path.exists(WRONG_CSV) or os.path.getsize(WRONG_CSV) == 0:
        with open(WRONG_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(WRONG_HEADERS)

_db_lock = threading.RLock()
_db = None

def _apply_sqlite_pragmas(conn):
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.close()
    except Exception:
        pass

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

# ------------------------------ 경보 ------------------------------
def _emit_alert(msg: str):
    try:
        print(f"🔴 [ALERT] {msg}")
        os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
        with open(ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{_now_kst_iso()} {msg}\n")
    except Exception:
        pass

# ------------------------------ 리더/헬퍼 ------------------------------
def _read_recent_failures_for(symbol: str, strategy: str, limit: int = 2000) -> pd.DataFrame:
    """same sym/strategy 최근 실패 일부만 로드(가벼운 유사도/샘플링용)"""
    if not os.path.exists(WRONG_CSV):
        return pd.DataFrame()
    use = ["timestamp","symbol","strategy","feature_hash","feature_vector","predicted_class","reason","return_value"]
    try:
        df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig", usecols=lambda c: c in use)
    except Exception:
        return pd.DataFrame()
    df = df[(df["symbol"]==symbol) & (df["strategy"]==strategy)].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=False)
    return df.head(limit)

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

# ------------------------------ 중복/폭주 가드 ------------------------------
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

        ts_min = (ts - pd.Timedelta(minutes=90)).isoformat()
        ts_max = (ts + pd.Timedelta(minutes=90)).isoformat()

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
                t = t.dt.tz_localize("Asia/Seoul", nonexistent="NaT", ambiguous="NaT")
                m = (t >= pd.to_datetime(ts_min)) & (t <= pd.to_datetime(ts_max))
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

def _strategy_cap(strategy: str) -> int:
    if strategy == "단기":
        return FAIL_CAP_SHORT
    if strategy == "중기":
        return FAIL_CAP_MID
    return FAIL_CAP_LONG  # 장기 및 기타

def _within_sampling_cap(symbol: str, strategy: str, now_ts: datetime.datetime) -> bool:
    """윈도우(FAIL_WIN_MINUTES) 안에서 동일 심볼·전략의 실패 개수가 CAP 이하인지 확인"""
    df = _read_recent_failures_for(symbol, strategy, limit=5000)
    if df.empty:
        return True
    cap = _strategy_cap(strategy)
    cutoff = now_ts - pd.Timedelta(minutes=FAIL_WIN_MINUTES)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cnt = int((df["timestamp"] >= cutoff).sum())
    return cnt < cap

# ------------------------------ 유사도/분류 ------------------------------
def _parse_feature_vector(v_any) -> np.ndarray:
    if isinstance(v_any, (list, tuple, np.ndarray)):
        try: return np.asarray(v_any, dtype=float).reshape(-1)
        except Exception: return np.array([], dtype=float)
    if isinstance(v_any, str) and v_any.strip().startswith("["):
        try: return np.asarray(json.loads(v_any), dtype=float).reshape(-1)
        except Exception: return np.array([], dtype=float)
    return np.array([], dtype=float)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _similarity_level(symbol: str, strategy: str, feature_vec: np.ndarray) -> Tuple[str, float]:
    """최근 실패들과 코사인 유사도 기반으로 recur/evo/noise 중 분류"""
    try:
        recent = _read_recent_failures_for(symbol, strategy, limit=2000)
        if recent.empty or feature_vec.size == 0:
            return ("evo", 0.0)

        feats: List[np.ndarray] = []
        for v in recent["feature_vector"].tolist():
            feats.append(_parse_feature_vector(v))
        feats = [f for f in feats if f.size == feature_vec.size and f.size > 0]
        if not feats:
            return ("evo", 0.0)

        sims: List[float] = []
        step = max(1, len(feats) // max(1, FAIL_SIM_TOPK))
        for i in range(0, len(feats), step):
            sims.append(_cosine_sim(feature_vec, feats[i]))
            if len(sims) >= FAIL_SIM_TOPK:
                break

        if not sims:
            return ("evo", 0.0)
        best = max(sims)

        if best >= FAIL_SIM_RECUR:
            return ("recur", best)
        if best >= FAIL_SIM_EVO:
            return ("evo", best)
        return ("noise", best)
    except Exception:
        return ("evo", 0.0)

def _auto_failure_reason(rec: Dict[str, Any]) -> str:
    try:
        lbl = rec.get("label", None)
        if lbl not in (None, ""):
            try:
                if int(lbl) < 0:
                    return "negative_label"
            except Exception:
                return "nan_label"
        rp = rec.get("raw_prob", None)
        cp = rec.get("calib_prob", None)
        def _is_bad(v):
            try:
                vv = float(v)
                return math.isnan(vv) or math.isinf(vv)
            except Exception:
                return False
        if _is_bad(rp) or _is_bad(cp):
            return "prob_nan"

        rs = str(rec.get("reason","")).strip().lower()
        if "class_out_of_range" in rs:
            return "class_out_of_range"
        if "bounds" in rs or "range" in rs:
            return "bounds_mismatch"
        return rs if rs else "unknown"
    except Exception:
        return "unknown"

def _compute_train_weight(strategy: str, level: str, ts: datetime.datetime) -> float:
    base = BASE_WEIGHT.get(strategy, BASE_WEIGHT["장기"]).get(level, 0.0)
    try:
        age_days = 0.0
        tau = 30.0
        decay = math.exp(-age_days / tau)
    except Exception:
        decay = 1.0
    return round(float(base * decay), 6)

def _is_noise_by_return(rv: Any) -> bool:
    try:
        v = float(rv)
        return abs(v) < FAIL_NOISE_MIN_RET
    except Exception:
        return False

# ------------------------------ CSV append (락/재시도) ------------------------------
def _append_wrong_csv_row(row: Dict[str, Any], max_retries: int = 5, sleep_sec: float = 0.05):
    _ensure_wrong_csv()
    attempt = 0
    while True:
        try:
            try:
                import fcntl
                with open(WRONG_CSV, "a", newline="", encoding="utf-8-sig") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    except Exception:
                        pass
                    w = csv.DictWriter(f, fieldnames=WRONG_HEADERS)
                    w.writerow({k: row.get(k, "") for k in WRONG_HEADERS})
                    try:
                        f.flush(); os.fsync(f.fileno())
                    except Exception:
                        pass
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

# ------------------------------ 메인 API ------------------------------
def insert_failure_record(record: Dict[str, Any],
                          feature_hash: Optional[str] = None,
                          label: Optional[int] = None,
                          feature_vector: Optional[Iterable[float]] = None,
                          context: Optional[str] = None) -> bool:
    """
    예측 실패/평가 실패 등 한 건을 기록.
    - 중복·폭주 가드를 통과해야 CSV/SQLite에 반영됨.
    - 반환값: 실제로 기록했으면 True, 스킵/오류면 False
    """
    try:
        ensure_failure_db()
        rec = _sanitize_dict(dict(record or {}))

        ts_iso = rec.get("timestamp") or _now_kst_iso()
        sym = str(rec.get("symbol","UNKNOWN"))
        strat = str(rec.get("strategy","알수없음"))
        pcls = _safe_int(rec.get("predicted_class"), default=-1)
        lbl  = label if label is not None else _safe_int(rec.get("label"), default=-1)

        fv = feature_vector if feature_vector is not None else rec.get("feature_vector", None)
        if isinstance(fv, str) and fv.strip().startswith("["):
            try: fv = json.loads(fv)
            except Exception: fv = []
        fh = feature_hash or rec.get("feature_hash") or (_sha1_of_list(fv) if isinstance(fv,(list,tuple,np.ndarray)) else "none")

        row = {
            "timestamp": ts_iso, "symbol": sym, "strategy": strat,
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

        auto_reason = _auto_failure_reason({**rec, **row})
        if not str(row["reason"]).strip():
            row["reason"] = auto_reason

        if check_failure_exists({**rec, **row}):
            return False

        feat_vec = _parse_feature_vector(row["feature_vector"])
        level, sim = _similarity_level(sym, strat, feat_vec)

        if level != "recur" and _is_noise_by_return(row.get("return_value", "")):
            level = "noise"

        row["failure_level"] = level
        row["train_weight"]  = _compute_train_weight(strat, level, _now_kst())

        if level == "noise":
            print(f"[failure_db] skip noise {sym}-{strat} pcls={row['predicted_class']} sim={sim:.3f}")
            return False

        if not _within_sampling_cap(sym, strat, _now_kst()):
            if level == "recur":
                print(f"[failure_db] drop(recur-cap) {sym}-{strat} pcls={row['predicted_class']}")
                return False
            if np.random.random() < 0.5:
                print(f"[failure_db] drop(evo-sample) {sym}-{strat} pcls={row['predicted_class']}")
                return False

        # CSV
        try:
            _append_wrong_csv_row(row)
        except Exception as e:
            print(f"[failure_db] CSV 기록 실패: {e}")

        # SQLite
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

# ------------------------------ 모듈 테스트 ------------------------------
if __name__ == "__main__":
    ensure_failure_db()
    demo = {
        "timestamp": _now_kst_iso(),
        "symbol": "BTCUSDT",
        "strategy": "장기",
        "predicted_class": 3,
        "label": 2,
        "model": "meta",
        "group_id": 0,
        "entry_price": 100.0,
        "target_price": 103.0,
        "return_value": 0.01,
        "reason": "",
        "context": "evaluation",
        "note": "",
        "regime": "unknown",
        "meta_choice": "test",
        "raw_prob": 0.21,
        "calib_prob": 0.19,
        "calib_ver": "v1",
        "feature_vector": [0.1, 0.2, 0.3, 0.4],
        "source": "평가",
        "source_exchange": "BYBIT",
    }
    ok = insert_failure_record(demo)
    print("inserted:", ok)
