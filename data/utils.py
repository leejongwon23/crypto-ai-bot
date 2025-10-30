# data/utils.py — 안정성/정합성 강화판 (MTF 피처화 + 경계보강/버킷균형 + augmented 소수클래스 증강 + 3단계 컨텍스트 + ✅ 온체인 컨텍스트)
# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}

import os, time, json, pickle, requests, pandas as pd, numpy as np, pytz, glob, hashlib, random
from sklearn.preprocessing import MinMaxScaler
from requests.exceptions import HTTPError, RequestException
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# === CHANGE [utils: quick label check] ===
def _count_valid_labels_for_df(df: pd.DataFrame, symbol: str, strategy: str) -> int:
    """
    Bybit만으로 라벨이 0개인지 빠르게 확인하기 위한 헬퍼.
    labels.make_labels가 있으면 실제 라벨 개수를, 없으면 -1을 반환.
    """
    try:
        if _make_labels is None or df is None or df.empty:
            return -1
        base = df[["timestamp", "close", "high", "low"]].copy()
        # timestamp 보정(있어도 안전)
        base["timestamp"] = _parse_ts_series(base["timestamp"])
        _, labels, *_ = _make_labels(base, symbol=symbol, strategy=strategy, group_id=None)
        return int(np.sum(labels >= 0))
    except Exception:
        return -1

# labels import 호환
try:
    from data.labels import make_labels as _make_labels
except Exception:
    try:
        from labels import make_labels as _make_labels
    except Exception:
        _make_labels = None

# 라벨 경계/그룹 config 일원화
from config import (
    get_class_ranges as cfg_get_class_ranges,
    get_class_groups as cfg_get_class_groups,
    get_NUM_CLASSES as cfg_get_NUM_CLASSES,
    get_EVAL_RUNTIME,
    MIN_FEATURES,
    # === CHANGE === CV, 경계 밴드 가져오기
    get_CV_CONFIG,
    BOUNDARY_BAND,
)

BASE_URL = "https://api.bybit.com"
BINANCE_BASE_URL = "https://fapi.binance.com"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuantWorker/1.0)"}
BINANCE_ENABLED = int(os.getenv("ENABLE_BINANCE", "1"))

# 예측 최소 윈도우
_PREDICT_MIN_WINDOW = int(os.getenv("PREDICT_WINDOW", "10"))

# ========================= 디렉터리/캐시 =========================
def _ensure_dir(path: str, fallback: str) -> str:
    try:
        os.makedirs(path, exist_ok=True); return path
    except OSError as e:
        if getattr(e, "errno", None) == 28:
            fb = os.path.join(fallback, os.path.basename(path).strip("/") or "dir")
            try:
                os.makedirs(fb, exist_ok=True)
                print(f"[⚠️ ENOSPC] '{path}' → '{fb}'"); return fb
            except Exception as e2:
                print(f"[❌ 폴백 실패] {fb}: {e2}")
        else:
            print(f"[⚠️ 디렉터리 생성 실패] {path}: {e}")
    return path

_CACHE_DIR = _ensure_dir(os.getenv("PRICE_CACHE_DIR", "/persistent/cache"), "/tmp")

def _cache_key(symbol: str, strategy: str, slack: int) -> str:
    return f"{symbol}__{strategy}__slack{int(slack)}.pkl"
def _cache_path(symbol: str, strategy: str, slack: int) -> str:
    return os.path.join(_CACHE_DIR, _cache_key(symbol, strategy, slack))
def _cache_ttl_seconds(interval: str) -> int:
    env = os.getenv("PRICE_CACHE_TTL_SEC")
    if env:
        try: return max(60, int(env))
        except Exception: pass
    m = {"60":600,"120":900,"240":1200,"360":1800,"720":1800,"D":3600,"W":6*3600,"M":24*3600}
    return m.get(str(interval), 1200)

def _load_df_cache(symbol: str, strategy: str, interval: str, slack: int):
    # 디스크 캐시: get_kline_by_strategy에서 메모리 캐시 미스 시 사용
    p = _cache_path(symbol, strategy, slack)
    if not os.path.exists(p): return None
    ttl = _cache_ttl_seconds(interval)
    try:
        if time.time() - os.path.getmtime(p) <= ttl:
            with open(p, "rb") as f:
                df = pickle.load(f)
                if isinstance(df, pd.DataFrame) and not df.empty: return df
    except Exception: pass
    return None

def _save_df_cache(symbol: str, strategy: str, slack: int, df: pd.DataFrame):
    try:
        with open(_cache_path(symbol, strategy, slack), "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as e:
        if getattr(e, "errno", None) == 28:
            print(f"[⚠️ ENOSPC] 캐시 저장 생략({_CACHE_DIR})")
        else:
            print(f"[⚠️ 캐시 저장 실패] {e}")
    except Exception: pass

def clear_price_cache(symbol: Optional[str] = None, strategy: Optional[str] = None):
    try:
        for fn in os.listdir(_CACHE_DIR):
            if not fn.endswith(".pkl"): continue
            if symbol and f"{symbol}__" not in fn: continue
            if strategy and f"__{strategy}__" not in fn: continue
            try: os.remove(os.path.join(_CACHE_DIR, fn))
            except Exception: pass
    except Exception: pass

# ========================= 외부 컨텍스트(선택적) =========================
try:
    from features.market import get_market_context_df as _get_market_ctx
except Exception:
    try:
        from market import get_market_context_df as _get_market_ctx
    except Exception:
        def _get_market_ctx(ts: pd.Series, strategy: str, symbol: Optional[str] = None) -> pd.DataFrame:
            return pd.DataFrame(columns=["timestamp"])
try:
    from features.correlations import get_rolling_corr_df as _get_corr_ctx
except Exception:
    try:
        from correlations import get_rolling_corr_df as _get_corr_ctx
    except Exception:
        def _get_corr_ctx(symbol: str, ts: pd.Series, strategy: str) -> pd.DataFrame:
            return pd.DataFrame(columns=["timestamp"])
try:
    from features.regime import get_regime_tags_df as _get_ext_regime_ctx
except Exception:
    try:
        from regime import get_regime_tags_df as _get_ext_regime_ctx
    except Exception:
        def _get_ext_regime_ctx(ts: pd.Series, strategy: str) -> pd.DataFrame:
            return pd.DataFrame(columns=["timestamp"])
try:
    from features.onchain import get_onchain_context_df as _get_onchain_ctx
except Exception:
    try:
        from onchain import get_onchain_context_df as _get_onchain_ctx
    except Exception:
        def _get_onchain_ctx(ts: pd.Series, strategy: str, symbol: Optional[str] = None) -> pd.DataFrame:
            return pd.DataFrame(columns=["timestamp"])

# ========================= 시간/머지 유틸 =========================
def _guess_tolerance_by_strategy(strategy: str) -> pd.Timedelta:
    iv = {"단기": pd.Timedelta(hours=2), "중기": pd.Timedelta(hours=12), "장기": pd.Timedelta(hours=12)}
    return iv.get(strategy, pd.Timedelta(hours=1))

def _parse_ts_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce", utc=True)
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            ts = s.copy()
            try:
                if getattr(ts.dt, "tz", None) is None: ts = ts.dt.tz_localize("UTC")
            except Exception: pass
            return ts.dt.tz_convert("Asia/Seoul")
        if pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            med = float(num.dropna().median()) if num.dropna().size else 0.0
            unit = "ms" if med >= 1e12 else "s"
            ts = pd.to_datetime(num, unit=unit, errors="coerce", utc=True)
            return ts.dt.tz_convert("Asia/Seoul")
        ss = s.astype(str).str.strip()
        is_digit_like = ss.str.match(r"^\d{10,13}$", na=False)
        if is_digit_like.mean() > 0.7:
            num = pd.to_numeric(ss.where(is_digit_like, np.nan), errors="coerce")
            med = float(num.dropna().median()) if num.dropna().size else 0.0
            unit = "ms" if med >= 1e12 else "s"
            ts = pd.to_datetime(num, unit=unit, errors="coerce", utc=True)
            return ts.dt.tz_convert("Asia/Seoul")
        ts = pd.to_datetime(ss, errors="coerce", utc=True)
        return ts.dt.tz_convert("Asia/Seoul")
    except Exception:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce", utc=True)

def _merge_asof_all(base: pd.DataFrame, add_list: List[pd.DataFrame], strategy: str) -> pd.DataFrame:
    out = base.copy()
    tol = _guess_tolerance_by_strategy(strategy)
    for add in add_list:
        if add is None or add.empty or "timestamp" not in add.columns: continue
        add = add.copy()
        add["timestamp"] = _parse_ts_series(add["timestamp"])
        cols = [c for c in add.columns if c != "timestamp"]
        if not cols: continue
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            add[["timestamp"] + cols].sort_values("timestamp"),
            on="timestamp", direction="backward", tolerance=tol,
        )
    return out

# --- 기본(백업) 심볼 시드 ---
_BASELINE_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","TRXUSDT",
    "LTCUSDT","BCHUSDT","LINKUSDT","ATOMUSDT","XLMUSDT","ETCUSDT","ICPUSDT","HBARUSDT","FILUSDT","SANDUSDT",
    "RNDRUSDT","INJUSDT","NEARUSDT","APEUSDT","ARBUSDT","AAVEUSDT","OPUSDT","SUIUSDT","DYDXUSDT","CHZUSDT",
    "LDOUSDT","STXUSDT","GMTUSDT","FTMUSDT","WLDUSDT","TIAUSDT","SEIUSDT","ARKMUSDT","JASMYUSDT","AKTUSDT",
    "GMXUSDT","SKLUSDT","BLURUSDT","ENSUSDT","CFXUSDT","FLOWUSDT","ALGOUSDT","MINAUSDT","NEOUSDT","MASKUSDT",
    "KAVAUSDT","BATUSDT","ZILUSDT","WAVESUSDT","OCEANUSDT","1INCHUSDT","YFIUSDT","STGUSDT","GALAUSDT","IMXUSDT"
]
SYMBOL_MAP = {"bybit": {}, "binance": {}}

try:
    from config import STRATEGY_CONFIG, get_REGIME, FEATURE_INPUT_SIZE as _FIS
except Exception:
    STRATEGY_CONFIG = {
        "단기": {"interval": "240", "limit": 1000, "binance_interval": "4h"},
        "중기": {"interval": "D", "limit": 500, "binance_interval": "1d"},
        "장기": {"interval": "D", "limit": 500, "binance_interval": "1d"},
    }
    def get_REGIME():
        return {"enabled": False, "atr_window": 14, "trend_window": 50, "vol_high_pct": 0.9, "vol_low_pct": 0.5}
    _FIS = 24

def _map_bybit_interval(interval: str) -> str:
    mapping = {"60":"60","1h":"60","120":"120","2h":"120","240":"240","4h":"240","360":"360","6h":"360",
               "720":"720","12h":"720","D":"D","1d":"D","W":"W","M":"M"}
    return mapping.get(str(interval), str(interval))

def _bybit_interval_minutes(mapped: str) -> int:
    m = str(mapped)
    if m.isdigit(): return int(m)
    if m=="D": return 1440
    if m=="W": return 10080
    if m=="M": return 43200
    return 60

# ========================= 그룹 순서/상태 =========================
_STATE_DIR = _ensure_dir("/persistent/state", "/tmp")
_STATE_PATH = os.path.join(_STATE_DIR, "group_order.json")
_STATE_BAK  = _STATE_PATH + ".bak"
_RUN_DIR = _ensure_dir("/persistent/run", "/tmp")
_PREDICT_GATE = os.path.join(_RUN_DIR, "predict_gate.json")
def _is_predict_gate_open() -> bool:
    try:
        if not os.path.exists(_PREDICT_GATE): return False
        with open(_PREDICT_GATE, "r", encoding="utf-8") as f:
            o = json.load(f)
        return bool(o.get("open", False))
    except Exception:
        return False

def _atomic_write_json(path: str, obj: dict):
    try: os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError as e:
        if getattr(e, "errno", None) == 28:
            print(f"[⚠️ ENOSPC] JSON 저장 생략({path})"); return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            try: f.flush(); os.fsync(f.fileno())
            except Exception: pass
        os.replace(tmp, path)
        try:
            dfd = os.open(os.path.dirname(path), os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        except Exception: pass
    except OSError as e:
        if getattr(e, "errno", None) == 28:
            print(f"[⚠️ ENOSPC] JSON 저장 생략({path})")
        else:
            print(f"[⚠️ JSON 저장 실패] {path}: {e}")
    except Exception: pass

class GroupOrderManager:
    def __init__(self, groups: List[List[str]]):
        self.groups = [list(g) for g in (groups[:8] if groups else [])]
        self.idx = 0
        self.trained = {}
        self.last_predicted_idx = -1
        self._load()
    def _load(self):
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            target = _STATE_PATH if os.path.isfile(_STATE_PATH) else (_STATE_BAK if os.path.isfile(_STATE_BAK) else None)
            if target:
                st = json.load(open(target, "r", encoding="utf-8"))
                saved_syms = st.get("symbols", [])
                saved_groups = _compute_groups(saved_syms, 5) if saved_syms else st.get("groups", [])
                if saved_groups: self.groups = saved_groups[:8]
                self.idx = int(st.get("idx", 0))
                self.trained = {int(k): set(v) for k, v in st.get("trained", {}).items()}
                self.last_predicted_idx = int(st.get("last_predicted_idx", -1))
                print(f"[🧭 그룹상태 로드] idx={self.idx}, last_predicted_idx={self.last_predicted_idx}")
        except Exception as e:
            print(f"[⚠️ 그룹상태 로드 실패] {e}")
    def _save(self):
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            payload = {
                "groups": self.groups, "idx": self.idx,
                "trained": {k: list(v) for k, v in self.trained.items()},
                "symbols": SYMBOLS, "last_predicted_idx": self.last_predicted_idx,
            }
            _atomic_write_json(_STATE_PATH, payload)
            _atomic_write_json(_STATE_BAK, payload)
        except Exception as e:
            print(f"[⚠️ 그룹상태 저장 실패] {e}")
    def current_index(self) -> int:
        return max(0, min(self.idx, max(0, len(self.groups) - 1)))
    def current_group(self) -> List[str]:
        i = self.current_index()
        return self.groups[i] if self.groups else []
    def _force_allow(self) -> bool:
        try:
            if os.getenv("TRAIN_FORCE_IGNORE_SHOULD", "0") == "1": return True
            return not _models_exist()
        except Exception:
            return False
    def should_train(self, symbol: str) -> bool:
        if self._force_allow():
            print(f"[order-override(utils)] {symbol}: force allow"); return True
        i = self.current_index(); gset = set(self.current_group()); done = self.trained.get(i, set())
        ok = (symbol in gset) and (symbol not in done)
        if not ok:
            where = "다음 그룹" if symbol not in gset else "이미 학습됨"
            print(f"[⛔ 순서강제] {symbol} → 현재 그룹{i} 아님 ({where})")
        try:
            gate = "open" if _is_predict_gate_open() else "closed"
            print(f"[order] group={i} gate={gate}")
        except Exception: pass
        return ok
    def mark_symbol_trained(self, symbol: str):
        i = self.current_index(); self.trained.setdefault(i, set()).add(symbol); self._save()
        print(f"[🧩 학습기록] 그룹{i}: {sorted(list(self.trained[i]))}/{self.current_group()}")
    def ready_for_group_predict(self) -> bool:
        i = self.current_index(); group = set(self.current_group()); done = self.trained.get(i, set())
        all_trained = group.issubset(done) and len(group) > 0; already_pred = (self.last_predicted_idx == i)
        if all_trained and not already_pred:
            print(f"[🚦 예측준비] 그룹{i} 완주"); return True
        if already_pred: print(f"[⏸ 보류] 그룹{i} 이미 예측 처리됨")
        else: print(f"[⏳ 대기] 미완료: {sorted(list(group-done))}")
        return False
    def mark_group_predicted(self):
        i = self.current_index()
        if self.last_predicted_idx == i:
            print(f"[🛡 중복차단] 그룹{i} 예측 완료 이미 반영"); return
        print(f"[✅ 예측완료] 그룹{i} → 다음 그룹"); self.last_predicted_idx = i
        self.idx = (i + 1) % max(1, len(self.groups)); self.trained.setdefault(self.idx, set()); self._save()
    def reset(self, start_index: int = 0):
        self.idx = max(0, min(start_index, max(0, len(self.groups) - 1)))
        self.trained = {self.idx: set()}; self.last_predicted_idx = -1; self._save()
        print(f"[♻️ 그룹순서 리셋] idx={self.idx}")
    def rebuild_groups(self, symbols: Optional[List[str]] = None, group_size: int = 5):
        self.groups = _compute_groups(symbols or SYMBOLS, group_size)[:8]; self.reset(0)
        print(f"[🧱 그룹재구성] → {len(self.groups)}그룹")

def _merge_unique(*lists):
    seen, out = set(), []
    for L in lists:
        for x in L:
            if x and x not in seen:
                seen.add(x); out.append(x)
    return out

def _discover_from_env():
    raw = os.getenv("PREDICT_SYMBOLS") or os.getenv("SYMBOLS_OVERRIDE") or ""
    if not raw.strip(): return []
    return [p.strip().upper() for p in raw.replace("\n", ",").replace(" ", ",").split(",") if p.strip()]

def _discover_from_models():
    md = "/persistent/models"
    if not os.path.isdir(md): return []
    syms = []
    for fn in os.listdir(md):
        if not (fn.endswith(".pt") or fn.endswith(".meta.json") or fn.endswith(".ptz")): continue
        sym = fn.split("_", 1)[0].upper()
        if sym.endswith("USDT") and len(sym) >= 6: syms.append(sym)
    return sorted(set(syms), key=syms.index)

def _select_60(symbols):
    if len(symbols) >= 40: return symbols[:40]
    need = 40 - len(symbols)
    filler = [s for s in _BASELINE_SYMBOLS if s not in symbols][:need]
    return symbols + filler

def _compute_groups(symbols, group_size=5):
    return [symbols[i:i+group_size] for i in range(0, len(symbols), group_size)]

# 고정 심볼/그룹
SYMBOLS = list(_BASELINE_SYMBOLS[:40])
SYMBOL_GROUPS = _compute_groups(SYMBOLS, 5)
SYMBOL_MAP["bybit"]  = {s: s for s in SYMBOLS}
SYMBOL_MAP["binance"] = {s: s for s in SYMBOLS}

def get_ALL_SYMBOLS(): return list(SYMBOLS)
def get_SYMBOL_GROUPS(): return list(SYMBOL_GROUPS)

GROUP_MGR = GroupOrderManager(SYMBOL_GROUPS)
def should_train_symbol(symbol: str) -> bool: return GROUP_MGR.should_train(symbol)
def mark_symbol_trained(symbol: str) -> None: GROUP_MGR.mark_symbol_trained(symbol)
def ready_for_group_predict() -> bool: return GROUP_MGR.ready_for_group_predict()
def mark_group_predicted() -> None: GROUP_MGR.mark_group_predicted()
def get_current_group_index() -> int: return GROUP_MGR.current_index()
def get_current_group_symbols() -> List[str]: return GROUP_MGR.current_group()
def reset_group_order(start_index: int = 0) -> None: GROUP_MGR.reset(start_index)
def rebuild_symbol_groups(symbols: Optional[List[str]] = None, group_size: int = 5) -> None: GROUP_MGR.rebuild_groups(symbols, group_size)
def group_all_complete() -> bool:
    i = GROUP_MGR.current_index(); group = set(GROUP_MGR.current_group()); done = GROUP_MGR.trained.get(i, set())
    return (len(group) > 0) and group.issubset(done)

def _models_exist(model_dir="/persistent/models"):
    try:
        if not os.path.isdir(model_dir): return False
        for fn in os.listdir(model_dir):
            full = os.path.join(model_dir, fn)
            if os.path.isdir(full):
                for _, _, files in os.walk(full):
                    if any(f.endswith((".pt", ".ptz", ".meta.json")) for f in files): return True
            else:
                if fn.endswith((".pt", ".ptz", ".meta.json")): return True
        return False
    except Exception:
        return False

class CacheManager:
    _cache = {}; _ttl = {}
    @classmethod
    def get(cls, key, ttl_sec=None):
        now = time.time()
        if key in cls._cache and (ttl_sec is None or now - cls._ttl.get(key, 0) < ttl_sec):
            return cls._cache[key]
        if key in cls._cache: cls.delete(key)
        return None
    @classmethod
    def set(cls, key, value):
        cls._cache[key] = value; cls._ttl[key] = time.time()
    @classmethod
    def delete(cls, key):
        if key in cls._cache:
            del cls._cache[key]; cls._ttl.pop(key, None)
    @classmethod
    def clear(cls):
        cls._cache.clear(); cls._ttl.clear(); print("[캐시 CLEAR ALL]")

def _auto_reset_group_state_if_needed():
    force = os.getenv("FORCE_RESET_GROUPS", "0") == "1"
    no_models = not _models_exist()
    if force or no_models:
        try:
            GROUP_MGR.reset(0); 
            try: CacheManager.clear()
            except Exception: pass
            print(f"[♻️ AUTO-RESET] groups(force={force}, no_models={no_models})")
        except Exception as e:
            print(f"[⚠️ AUTO-RESET 실패] {e}")
_auto_reset_group_state_if_needed()

# ========================= 캐시/백오프 =========================
def _binance_blocked_until(): return CacheManager.get("binance_blocked_until")
def _is_binance_blocked(): 
    u = _binance_blocked_until(); return u is not None and time.time() < u
def _block_binance_for(seconds=1800):
    CacheManager.set("binance_blocked_until", time.time() + seconds)
    print(f"[🚫 Binance 차단] {seconds}s")
def _bybit_blocked_until(): return CacheManager.get("bybit_blocked_until")
def _is_bybit_blocked():
    u = _bybit_blocked_until(); return u is not None and time.time() < u
def _block_bybit_for(seconds=900):
    CacheManager.set("bybit_blocked_until", time.time() + seconds)
    print(f"[🚫 Bybit 차단] {seconds}s")

# ========================= 실패 로깅 경량 헬퍼 =========================
def safe_failed_result(symbol, strategy, reason=""):
    try:
        from failure_db import insert_failure_record
        payload = {
            "symbol": symbol or "UNKNOWN", "strategy": strategy or "UNKNOWN",
            "model": "utils", "reason": reason,
            "timestamp": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_class": -1, "label": -1
        }
        insert_failure_record(payload, feature_vector=[])
    except Exception as e:
        print(f"[⚠️ safe_failed_result 실패] {e}")

# ========================= 기타 유틸 =========================
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}
def get_btc_dominance():
    global BTC_DOMINANCE_CACHE
    now = time.time()
    if now - BTC_DOMINANCE_CACHE["timestamp"] < 1800:
        return BTC_DOMINANCE_CACHE["value"]
    try:
        res = requests.get("https://api.coinpaprika.com/v1/global", timeout=10, headers=REQUEST_HEADERS)
        res.raise_for_status()
        dom = float(res.json().get("bitcoin_dominance_percentage", 50.0)) / 100.0
        BTC_DOMINANCE_CACHE = {"value": round(dom, 4), "timestamp": now}
        return BTC_DOMINANCE_CACHE["value"]
    except Exception:
        return BTC_DOMINANCE_CACHE["value"]

def future_gains_by_hours(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df is None or len(df) == 0 or "timestamp" not in df.columns:
        return np.zeros(0 if df is None else len(df), dtype=np.float32)
    ts = _parse_ts_series(df["timestamp"])
    close = pd.to_numeric(df["close"], errors="coerce").astype(np.float32).values
    high  = pd.to_numeric((df["high"] if "high" in df.columns else df["close"]), errors="coerce").astype(np.float32).values
    out = np.zeros(len(df), dtype=np.float32)
    H = pd.Timedelta(hours=int(horizon_hours)); j0 = 0
    for i in range(len(df)):
        t0 = ts.iloc[i]; t1 = t0 + H
        j = max(j0, i); mx = high[i]
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > mx: mx = high[j]
            j += 1
        j0 = max(j - 1, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        out[i] = float((mx - base) / (base + 1e-12))
    return out.astype(np.float32)

def future_gains(df: pd.DataFrame, strategy: str) -> np.ndarray:
    return future_gains_by_hours(df, {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24))

def _downcast_numeric(df: pd.DataFrame, prefer_float32: bool = True) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    for c in df.columns:
        try:
            if pd.api.types.is_integer_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], downcast="integer")
            elif pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], downcast="float")
                if prefer_float32 and df[c].dtype == np.float64: df[c] = df[c].astype(np.float32)
        except Exception: pass
    return df

def _fix_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        else: df[c] = np.nan
    df[["open","high","low","close"]] = df[["open","high","low","close"]].replace([np.inf, -np.inf], np.nan)
    df["volume"] = df["volume"].replace([np.inf, -np.inf], np.nan)
    df.loc[df[["open","high","low","close"]].le(0).any(axis=1), ["open","high","low","close"]] = np.nan
    df.loc[df["volume"] < 0, "volume"] = np.nan
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].ffill()
    df = df.dropna(subset=["open","high","low","close","volume"])
    mx = df[["open","close","low"]].max(axis=1); mn = df[["open","close","high"]].min(axis=1)
    df["high"] = np.maximum(df["high"].values, mx.values)
    df["low"]  = np.minimum(df["low"].values,  mn.values)
    return df

def _winsorize_prices(df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    for c in ["open","high","low","close"]:
        if c not in df.columns: continue
        s = pd.to_numeric(df[c], errors="coerce")
        lo = s.quantile(lower_q); hi = s.quantile(upper_q)
        df[c] = s.clip(lower=lo, upper=hi).astype(np.float32)
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce")
        lo = max(0.0, v.quantile(lower_q)); hi = v.quantile(upper_q)
        df["volume"] = v.clip(lower=lo, upper=hi).astype(np.float32)
    return df

# ========================= 라벨링 유틸 =========================
def _label_with_edges(values: np.ndarray, edges: List[Tuple[float, float]]) -> np.ndarray:
    if len(edges) <= 1: return np.zeros(len(values), dtype=np.int64)
    stops = np.array([b for (_, b) in edges[:-1]], dtype=np.float64)
    idx = np.digitize(values, stops, right=True)
    return np.clip(idx, 0, len(edges)-1).astype(np.int64)

# ========================= 거래소/수집 =========================
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp","open","high","low","close","volume","datetime"]
    if df is None: return pd.DataFrame(columns=cols)
    df = df.copy()
    ts_candidate = None
    for k in ["timestamp","start","t","open_time","openTime","time",0]:
        if (isinstance(k,int) and k in getattr(df,"columns",[])) or (isinstance(k,str) and k in df.columns):
            ts_candidate = k; break
    if ts_candidate is not None: df["timestamp"] = _parse_ts_series(df[ts_candidate])
    else: df["timestamp"] = _parse_ts_series(pd.Series([pd.NaT]*len(df)))
    ren = {}
    if "1" in df.columns: ren["1"]="open"
    if "2" in df.columns: ren["2"]="high"
    if "3" in df.columns: ren["3"]="low"
    if "4" in df.columns: ren["4"]="close"
    if "5" in df.columns: ren["5"]="volume"
    if ren: df = df.rename(columns=ren)
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        else: df[c] = np.nan
    df = df.dropna(subset=["timestamp","open","high","low","close","volume"])
    df["datetime"] = df["timestamp"]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = _downcast_numeric(df); df = _fix_ohlc_consistency(df); df = _winsorize_prices(df, 0.001, 0.999)
    try: df["timestamp"] = _parse_ts_series(df["timestamp"])
    except Exception: pass
    df = df.dropna(subset=["timestamp","open","high","low","close","volume"]).reset_index(drop=True)
    return df[cols]

def _clip_tail(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df is None or df.empty: return df
    if len(df) > limit: df = df.iloc[-limit:].reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    except Exception: pass
    mask = ts.diff().fillna(pd.Timedelta(seconds=0)) >= pd.Timedelta(seconds=0)
    if not mask.all(): df = df[mask].reset_index(drop=True)
    return df

# Bybit
def get_kline(symbol: str, interval: str = "60", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    if _is_bybit_blocked():
        print("[⛔ Bybit 비활성화]"); return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])
    real_symbol = SYMBOL_MAP["bybit"].get(symbol, symbol)
    target_rows = int(limit); collected, total, last_oldest = [], 0, None
    interval = _map_bybit_interval(interval); iv_minutes = _bybit_interval_minutes(interval)
    start_ms = None
    if end_time is None:
        lookback_ms = int(target_rows * iv_minutes * 60 * 1000)
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        start_ms = max(0, now_ms - lookback_ms)
    empty_resp_count = 0
    while total < target_rows:
        success = False
        for _ in range(max_retry):
            try:
                rows_needed = target_rows - total; req = min(1000, rows_needed)
                for category in ("linear","spot"):
                    params = {"category": category, "symbol": real_symbol, "interval": interval, "limit": req}
                    if end_time is not None: params["end"] = int(end_time.timestamp() * 1000)
                    elif start_ms is not None: params["start"] = start_ms
                    res = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10, headers=REQUEST_HEADERS)
                    res.raise_for_status(); data = res.json()
                    raw = (data or {}).get("result", {}).get("list", [])
                    if not raw:
                        empty_resp_count += 1; continue
                    if isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6:
                        df_chunk = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
                    else:
                        df_chunk = pd.DataFrame(raw)
                    df_chunk = _normalize_df(df_chunk)
                    if df_chunk.empty: continue
                    collected.append(df_chunk); total += len(df_chunk); success = True
                    oldest_ts = df_chunk["timestamp"].min()
                    if last_oldest is not None and pd.to_datetime(oldest_ts) >= pd.to_datetime(last_oldest):
                        oldest_ts = pd.to_datetime(oldest_ts) - pd.Timedelta(minutes=1)
                    last_oldest = oldest_ts
                    end_time = pd.to_datetime(oldest_ts).tz_convert("UTC") - pd.Timedelta(milliseconds=1)
                    time.sleep(0.2); break
                if success: break
            except RequestException:
                time.sleep(1); continue
            except Exception:
                time.sleep(0.5); continue
        if not success: break
    if collected:
        df = _normalize_df(pd.concat(collected, ignore_index=True)); df.attrs["source_exchange"] = "BYBIT"; return df
    if empty_resp_count >= max_retry * 2: _block_bybit_for(int(os.getenv("BYBIT_BACKOFF_SEC", "900")))
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# Binance
def get_kline_binance(symbol: str, interval: str = "240", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    real_symbol = SYMBOL_MAP["binance"].get(symbol, symbol)
    _bin_iv = None
    for _, cfg in STRATEGY_CONFIG.items():
        if cfg.get("interval") == interval: _bin_iv = cfg.get("binance_interval"); break
    if _bin_iv is None: _bin_iv = {"240":"4h","D":"1d","2D":"2d","60":"1h"}.get(interval, "1h")
    target_rows = int(limit); collected, total, last_oldest = [], 0, None
    if not BINANCE_ENABLED or _is_binance_blocked():
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])
    while total < target_rows:
        success = False
        for _ in range(max_retry):
            try:
                rows_needed = target_rows - total; req = min(1000, rows_needed)
                params = {"symbol": real_symbol, "interval": _bin_iv, "limit": req}
                if end_time is not None: params["endTime"] = int(end_time.timestamp() * 1000)
                res = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/klines", params=params, timeout=10, headers=REQUEST_HEADERS)
                try: res.raise_for_status()
                except HTTPError as he:
                    if getattr(he.response, "status_code", None) == 418:
                        _block_binance_for(1800)
                        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])
                    raise
                raw = res.json()
                if not raw: break
                if isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6:
                    df_chunk = pd.DataFrame(raw, columns=[
                        "timestamp","open","high","low","close","volume",
                        "close_time","quote_asset_volume","trades","taker_base_vol","taker_quote_vol","ignore"
                    ])
                else:
                    df_chunk = pd.DataFrame(raw)
                df_chunk = _normalize_df(df_chunk)
                if df_chunk.empty: break
                collected.append(df_chunk); total += len(df_chunk); success = True
                if total >= target_rows: break
                oldest_ts = df_chunk["timestamp"].min()
                if last_oldest is not None and pd.to_datetime(oldest_ts) >= pd.to_datetime(last_oldest):
                    oldest_ts = pd.to_datetime(oldest_ts) - pd.Timedelta(minutes=1)
                last_oldest = oldest_ts
                end_time = pd.to_datetime(oldest_ts).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)
                time.sleep(0.3); break
            except RequestException:
                time.sleep(1); continue
            except Exception:
                time.sleep(0.5); continue
        if not success: break
    if collected:
        df = _normalize_df(pd.concat(collected, ignore_index=True)); df.attrs["source_exchange"] = "BINANCE"; return df
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 통합 수집 + 병합 =========================
def get_merged_kline_by_strategy(symbol: str, strategy: str) -> pd.DataFrame:
    config = STRATEGY_CONFIG.get(strategy)
    if not config: return pd.DataFrame()
    interval = config["interval"]; base_limit = int(config["limit"]); max_total = base_limit
    def fetch_until_target(fetch_func):
        total = []; end = None; cnt = 0; max_rep = 10
        while cnt < max_total and len(total) < max_rep:
            dfc = fetch_func(symbol, interval=interval, limit=base_limit, end_time=end)
            if dfc is None or dfc.empty: break
            total.append(dfc); cnt += len(dfc)
            if len(dfc) < base_limit: break
            oldest = dfc["timestamp"].min()
            end = pd.to_datetime(oldest).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)
        return _normalize_df(pd.concat(total, ignore_index=True)) if total else pd.DataFrame()
    df_bybit = fetch_until_target(get_kline)
    df_binance = fetch_until_target(get_kline_binance) if len(df_bybit) < base_limit and BINANCE_ENABLED and not _is_binance_blocked() else pd.DataFrame()
    df_all = _normalize_df(pd.concat([df_bybit, df_binance], ignore_index=True)) if (not df_bybit.empty or not df_binance.empty) else pd.DataFrame()
    if df_all.empty: return pd.DataFrame()
    df_all = _clip_tail(df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), base_limit)
    srcs = []; 
    if not df_bybit.empty: srcs.append("BYBIT")
    if not df_binance.empty: srcs.append("BINANCE")
    df_all.attrs["source_exchange"] = "+".join(srcs) if srcs else "UNKNOWN"
    for c in ["timestamp","open","high","low","close","volume"]:
        if c not in df_all.columns:
            df_all[c] = 0.0 if c != "timestamp" else pd.Timestamp.now(tz="Asia/Seoul")
    df_all.attrs["augment_needed"] = len(df_all) < base_limit
    return df_all

# =============== 임의 인터벌 수집기(MTF) ===============
def get_kline_interval(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        df_bybit = get_kline(symbol, interval=interval, limit=limit)
        if (df_bybit is None or df_bybit.empty) and not _is_binance_blocked():
            df_bin = get_kline_binance(symbol, interval=interval, limit=limit)
            return _normalize_df(df_bin)
        return _normalize_df(df_bybit)
    except Exception:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

def get_kline_by_strategy(symbol: str, strategy: str, end_slack_min: int = 0):
    """
    거래소 통합 수집 → 정규화 → 슬랙 컷 → 연속구간 보장 → 한도(limit) 정확히 맞춤.
    + 추가 규칙(요청사항 반영):
      1) Bybit만으로 라벨이 0개일 경우 → Binance를 강제 병합 후 재구성.
      2) 장기(주봉/일봉 기반)인데 Bybit 캔들이 부족하면 → 조건 없이 Binance 병합.
    """
    try:
        # 슬랙 기본값
        if not end_slack_min:
            try:
                end_slack_min = int(get_EVAL_RUNTIME().get("price_window_slack_min", 10))
            except Exception:
                end_slack_min = 0

        # 전략 파라미터
        cfg = STRATEGY_CONFIG.get(strategy, {"limit": 300, "interval": "D"})
        limit = int(cfg.get("limit", 300))
        interval = cfg.get("interval", "D")

        # 메모리/디스크 캐시
        cache_key = f"{symbol.upper()}-{strategy}-slack{end_slack_min}"
        cached = CacheManager.get(cache_key, ttl_sec=600)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        cached_disk = _load_df_cache(symbol, strategy, interval, end_slack_min)
        if isinstance(cached_disk, pd.DataFrame) and not cached_disk.empty:
            CacheManager.set(cache_key, cached_disk)
            return cached_disk

        # 1) 원천 수집(Bybit 우선)
        df_bybit = get_kline(symbol, interval=interval, limit=limit)
        if not isinstance(df_bybit, pd.DataFrame):
            df_bybit = pd.DataFrame()

        # 2) 장기일 때 무조건 Binance 병합 트리거
        force_long_merge = (strategy == "장기") and (len(df_bybit) < limit)
        df_binance = pd.DataFrame()

        # 3) 기본 병합 조건(부족 시) 또는 장기 강제 병합
        if (df_bybit.empty or len(df_bybit) < int(limit * 0.9) or force_long_merge) and BINANCE_ENABLED and not _is_binance_blocked():
            df_binance = get_kline_binance(symbol, interval=interval, limit=limit)

        # 4) 1차 병합
        dfs = [d for d in [df_bybit, df_binance] if isinstance(d, pd.DataFrame) and not d.empty]
        df = _normalize_df(pd.concat(dfs, ignore_index=True)) if dfs else pd.DataFrame()

        # 5) Bybit만으로 충분해 보이지만, 라벨이 0개라면 Binance를 추가 병합
        if (df_binance is None or df_binance.empty) and (not df_bybit.empty) and BINANCE_ENABLED and not _is_binance_blocked():
            # 빠른 라벨 개수 점검(가능하면)
            valid_cnt = _count_valid_labels_for_df(df_bybit, symbol, strategy)
            if valid_cnt == 0:
                try:
                    add_bin = get_kline_binance(symbol, interval=interval, limit=limit)
                    if isinstance(add_bin, pd.DataFrame) and not add_bin.empty:
                        df = _normalize_df(pd.concat([df_bybit, add_bin], ignore_index=True))
                except Exception:
                    pass

        # 6) 최소 보정(완전 빈 경우 더미 1행)
        if df.empty:
            now = pd.Timestamp.now(tz="Asia/Seoul")
            df = pd.DataFrame(
                {"timestamp": [now], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0], "datetime": [now]}
            )
            df.attrs["source_exchange"] = "DUMMY"

        # 7) 슬랙 컷: 최신 n분 제거
        if end_slack_min > 0 and "timestamp" in df.columns and not df.empty:
            ts = _parse_ts_series(df["timestamp"])
            cutoff = ts.max() - pd.Timedelta(minutes=int(end_slack_min))
            df = df.loc[ts <= cutoff].copy()

        # 8) 숫자 컬럼 보정 및 NaN/inf 제거
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["timestamp", "open", "high", "low", "volume", "close"], inplace=True)

        # 9) 시간 정렬과 연속구간 보장
        if not df.empty:
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # 10) limit 정확히 맞추기
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)

        # 11) 연속성 체크 후 파손 구간 제거
        if not df.empty:
            ts = _parse_ts_series(df["timestamp"])
            diffs = ts.diff().dt.total_seconds().fillna(0)
            df = df.loc[diffs >= 0].reset_index(drop=True)

        # 12) 메타 정보
        srcs = []
        if not df_bybit.empty:   srcs.append("BYBIT")
        if 'df_binance' in locals() and isinstance(df_binance, pd.DataFrame) and not df_binance.empty: srcs.append("BINANCE")
        if "source_exchange" not in df.attrs or not df.attrs["source_exchange"]:
            df.attrs["source_exchange"] = "+".join(srcs) if srcs else "UNKNOWN"

        df.attrs["recent_rows"] = int(len(df))
        df.attrs["augment_needed"] = len(df) < limit
        df.attrs["enough_for_training"] = len(df) >= int(limit * 0.9)
        df.attrs["not_enough_rows"] = len(df) < _PREDICT_MIN_WINDOW

        # 13) 캐시 저장
        CacheManager.set(cache_key, df)
        _save_df_cache(symbol, strategy, end_slack_min, df)
        return df

    except Exception as e:
        print(f"[❌ get_kline_by_strategy 실패] {symbol}/{strategy}: {e}")
        safe_failed_result(symbol, strategy, reason=str(e))
        now = pd.Timestamp.now(tz="Asia/Seoul")
        df = pd.DataFrame(
            {"timestamp": [now], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0], "datetime": [now]}
        )
        df.attrs["source_exchange"] = "DUMMY"
        df.attrs["recent_rows"] = 1
        df.attrs["augment_needed"] = True
        df.attrs["enough_for_training"] = False
        df.attrs["not_enough_rows"] = True
        return df

# ========================= 프리패치/티커 =========================
def prefetch_symbol_groups(strategy: str):
    for group in SYMBOL_GROUPS:
        for sym in group:
            try: get_kline_by_strategy(sym, strategy)
            except Exception: pass

def get_realtime_prices():
    url = f"{BASE_URL}/v5/market/tickers"; params = {"category": "linear"}
    try:
        res = requests.get(url, params=params, timeout=10, headers=REQUEST_HEADERS)
        res.raise_for_status(); data = res.json()
        if "result" not in data or "list" not in data["result"]: return {}
        tickers = data["result"]["list"]; symset = set(get_ALL_SYMBOLS())
        return {item["symbol"]: float(item["lastPrice"]) for item in tickers if item["symbol"] in symset}
    except Exception:
        return {}

# ========================= 피처 생성(MTF + 레짐 + 컨텍스트) =========================
def _ensure_columns(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns: df[c] = 0.0

def _macd_parts(close: pd.Series, span_fast=12, span_slow=26, span_sig=9):
    ema_fast = close.ewm(span=span_fast, adjust=False).mean()
    ema_slow = close.ewm(span=span_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=span_sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def _bbands(close: pd.Series, window=20):
    ma = close.rolling(window=window, min_periods=1).mean()
    sd = close.rolling(window=window, min_periods=1).std()
    upper = ma + 2*sd
    lower = ma - 2*sd
    width = (upper - lower) / (ma + 1e-6)
    percent_b = (close - lower) / ((upper - lower) + 1e-6)
    return ma, upper, lower, sd, width, percent_b

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_window=14, d_window=3):
    lowest = low.rolling(k_window, min_periods=1).min()
    highest = high.rolling(k_window, min_periods=1).max()
    k = 100 * (close - lowest) / ((highest - lowest) + 1e-6)
    d = k.rolling(d_window, min_periods=1).mean()
    return k, d

def _prefix_cols(df: pd.DataFrame, prefix: str, skip=("timestamp",)) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy(); ren = {c: f"{prefix}_{c}" for c in df.columns if c not in skip}
    return df.rename(columns=ren)

def _compute_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns: df[c] = 0.0
    df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))
    macd, macd_sig, macd_hist = _macd_parts(df["close"])
    df["macd"] = macd; df["macd_signal"] = macd_sig; df["macd_hist"] = macd_hist
    _, bb_up, bb_dn, bb_sd, bb_width, bb_pb = _bbands(df["close"], window=20)
    df["bb_up"] = bb_up; df["bb_dn"] = bb_dn; df["bb_sd"] = bb_sd
    df["bb_width"] = bb_width; df["bb_percent_b"] = bb_pb
    df["volatility"] = (df["high"] - df["low"]).abs()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["trend_score"] = (df["close"] > df["ema50"]).astype(np.float32)
    df["roc"] = df["close"].pct_change(periods=10)
    df["atr"] = _atr(df["high"], df["low"], df["close"], window=14)
    st_k, st_d = _stoch(df["high"], df["low"], df["close"], k_window=14, d_window=3)
    df["stoch_k"] = st_k; df["stoch_d"] = st_d
    highest14 = df["high"].rolling(14, min_periods=1).max()
    lowest14  = df["low"].rolling(14, min_periods=1).min()
    df["williams_r"] = -100 * (highest14 - df["close"]) / ((highest14 - lowest14) + 1e-6)
    try:
        import ta as _ta
        if "adx" not in df.columns:
            df["adx"] = _ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
        if "cci" not in df.columns:
            df["cci"] = _ta.trend.cci(df["high"], df["low"], df["close"], window=20, fillna=True)
        if "mfi" not in df.columns:
            df["mfi"] = _ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14, fillna=True)
        if "obv" not in df.columns:
            df["obv"] = _ta.volume.on_balance_volume(df["close"], df["volume"], fillna=True)
    except Exception:
        pass
    df["vwap"] = (df["volume"] * df["close"]).cumsum() / (df["volume"].cumsum() + 1e-6)
    return df

def _mtf_plan(strategy: str):
    plan = {"단기": ("60", ["240","D"]), "중기": ("240", ["D","3D"]), "장기": ("D", ["W","720"])}
    return plan.get(strategy, ("240", ["D"]))

def _synthesize_multi(df_base: pd.DataFrame, target_iv: str) -> pd.DataFrame:
    if df_base is None or df_base.empty: return pd.DataFrame()
    df = df_base.copy()
    df["timestamp"] = _parse_ts_series(df["timestamp"]).dt.tz_convert("UTC")
    df = df.set_index("timestamp")
    if target_iv not in {"3D","2D"}: return pd.DataFrame()
    rule = target_iv
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum(min_count=1)
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna(how="any").reset_index()
    out["timestamp"] = out["timestamp"].dt.tz_convert("Asia/Seoul"); out["datetime"] = out["timestamp"]
    return _normalize_df(out)

def _compute_mtf_features(symbol: str, strategy: str, df_base: pd.DataFrame) -> pd.DataFrame:
    _, aux_ivs = _mtf_plan(strategy)
    df_b = df_base.copy(); ctx_blocks = []
    for iv in aux_ivs:
        d = _synthesize_multi(df_b, iv) if iv in ("3D","2D") else get_kline_interval(symbol, iv, limit=max(200, len(df_b)))
        if d is None or d.empty: continue
        feat = _compute_feature_block(d)
        feat = feat[["timestamp","close","rsi","macd","macd_signal","bb_width","atr","ema20","ema50","ema100","ema200","stoch_k","stoch_d","williams_r","roc","trend_score","vwap"]]
        ctx_blocks.append(_prefix_cols(feat, f"f{iv}"))
    base_feat = _compute_feature_block(df_b)
    base_feat = base_feat[["timestamp","open","high","low","close","volume","rsi","macd","macd_signal","macd_hist","bb_up","bb_dn","bb_sd","bb_width","bb_percent_b","volatility","ema20","ema50","ema100","ema200","trend_score","roc","atr","stoch_k","stoch_d","williams_r","vwap"]]
    return _merge_asof_all(base_feat, ctx_blocks, strategy)

def compute_features(symbol: str, df: pd.DataFrame, strategy: str, required_features: list = None, fallback_input_size: int = None) -> pd.DataFrame:
    cache_key = f"{symbol}-{strategy}-features"
    cached = CacheManager.get(cache_key, ttl_sec=600)
    if cached is not None: return cached
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        safe_failed_result(symbol, strategy, reason="입력DataFrame empty"); dummy = pd.DataFrame(); dummy.attrs["not_enough_rows"] = True; return dummy
    if len(df) < _PREDICT_MIN_WINDOW:
        dummy = pd.DataFrame(); dummy.attrs["not_enough_rows"] = True; dummy.attrs["recent_rows"] = int(len(df)); return dummy
    df = df.copy()
    if "datetime" in df.columns: df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns: df["timestamp"] = pd.to_datetime("now", utc=True).tz_convert("Asia/Seoul")
    df["strategy"] = strategy
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns: df[c] = 0.0
    df = df[["timestamp","open","high","low","close","volume"]]
    if len(df) < 20:
        safe_failed_result(symbol, strategy, reason=f"row 부족 {len(df)}")
        dummy = pd.DataFrame(); dummy.attrs["not_enough_rows"] = True; dummy.attrs["recent_rows"] = int(len(df)); return dummy
    try:
        feat = _compute_mtf_features(symbol, strategy, df)
        regime_cfg = get_REGIME()
        if regime_cfg.get("enabled", False):
            try:
                atr_win = int(regime_cfg.get("atr_window", 14))
                trend_win = int(regime_cfg.get("trend_window", 50))
                vol_high = float(regime_cfg.get("vol_high_pct", 0.9))
                vol_low  = float(regime_cfg.get("vol_low_pct", 0.5))
                feat["atr_val"] = _atr(feat["high"], feat["low"], feat["close"], window=atr_win)
                thr_high = feat["atr_val"].quantile(vol_high); thr_low  = feat["atr_val"].quantile(vol_low)
                feat["vol_regime"] = np.where(feat["atr_val"] >= thr_high, 2, np.where(feat["atr_val"] <= thr_low, 0, 1))
                feat["ma_trend"] = feat["close"].rolling(window=trend_win, min_periods=1).mean()
                slope = feat["ma_trend"].diff()
                feat["trend_regime"] = np.where(slope > 0, 2, np.where(slope < 0, 0, 1))
                feat["regime_tag"] = feat["vol_regime"] * 3 + feat["trend_regime"]
            except Exception: pass
        try:
            ts = _parse_ts_series(feat["timestamp"])
            feat = _merge_asof_all(feat, [
                _get_market_ctx(ts, strategy, symbol),
                _get_corr_ctx(symbol, ts, strategy),
                _get_ext_regime_ctx(ts, strategy),
                _get_onchain_ctx(ts, strategy, symbol),
            ], strategy)
        except Exception: pass
        must_have = ["rsi","macd","macd_signal","macd_hist","ema20","ema50","ema100","ema200","bb_width","bb_percent_b","atr","stoch_k","stoch_d","williams_r","volatility","roc","vwap","trend_score"]
        _ensure_columns(feat, must_have)
        feat.replace([np.inf, -np.inf], 0, inplace=True); feat.fillna(0, inplace=True)
        feat_cols = [c for c in feat.columns if c != "timestamp"]
        if len(feat_cols) < _FIS:
            for i in range(len(feat_cols), _FIS):
                pad = f"pad_{i}"; feat[pad] = 0.0; feat_cols.append(pad)
        feat[feat_cols] = _downcast_numeric(feat[feat_cols]).astype(np.float32)
        feat[feat_cols] = MinMaxScaler().fit_transform(feat[feat_cols])
    except Exception as e:
        safe_failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        dummy = pd.DataFrame(); dummy.attrs["not_enough_rows"] = True; return dummy
    if feat.empty or feat.isnull().values.any():
        safe_failed_result(symbol, strategy, reason="최종 결과 DataFrame 오류")
        dummy = pd.DataFrame(); dummy.attrs["not_enough_rows"] = True; return dummy
    CacheManager.set(cache_key, feat); return feat

def compute_features_multi(symbol: str, df_base: pd.DataFrame) -> Dict[str, Optional[pd.DataFrame]]:
    out: Dict[str, Optional[pd.DataFrame]] = {"단기": None, "중기": None, "장기": None}
    try:
        if isinstance(df_base, pd.DataFrame) and not df_base.empty:
            f_short = compute_features(symbol, df_base, "단기")
            out["단기"] = f_short if isinstance(f_short, pd.DataFrame) and not f_short.empty else None
    except Exception:
        out["단기"] = None
    for strat in ("중기","장기"):
        try:
            df_s = get_kline_by_strategy(symbol, strat)
            if isinstance(df_s, pd.DataFrame) and not df_s.empty:
                f_s = compute_features(symbol, df_s, strat)
                out[strat] = f_s if isinstance(f_s, pd.DataFrame) and not f_s.empty else None
            else:
                out[strat] = None
        except Exception:
            out_strat = None
            out[strat] = out_strat
    return out

# ========================= 증강/중복 컷 =========================
def augment_jitter(seq: np.ndarray, sigma_min: float = 0.0005, sigma_max: float = 0.002) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.size == 0: return seq.copy()
    sigma = float(np.random.uniform(sigma_min, sigma_max))
    noise = np.random.normal(loc=0.0, scale=sigma, size=seq.shape).astype(np.float32)
    return (seq * (1.0 + noise)).astype(np.float32)

def augment_time_shift(seq: np.ndarray, max_shift: int = 2) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] <= 1 or max_shift <= 0: return seq.copy()
    shift = int(np.random.randint(-max_shift, max_shift + 1))
    if shift == 0: return seq.copy()
    w, f = seq.shape
    if shift > 0:
        pad = np.repeat(seq[0:1, :], shift, axis=0); new = np.vstack([pad, seq[:w-shift, :]])
    else:
        s = -shift; pad = np.repeat(seq[-1:, :], s, axis=0); new = np.vstack([seq[s:, :], pad])
    if new.shape[0] != w:
        if new.shape[0] > w: new = new[:w, :]
        else: new = np.vstack([new, np.repeat(seq[-1:, :], w - new.shape[0], axis=0)])
    return new.astype(np.float32)

def augment_for_min_count(X: np.ndarray, y: np.ndarray, target_count: int) -> Tuple[np.ndarray, np.ndarray]:
    if X is None or y is None: return X, y
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int64)
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    to_add, to_add_labels = [], []
    cap_total = max(X.shape[0] * 3, target_count * len(unique))
    for cls in unique:
        cur = class_counts.get(int(cls), 0)
        if cur >= target_count: continue
        need = target_count - cur
        idxs = np.where(y == cls)[0]
        if idxs.size == 0: continue
        gen = 0; attempts = 0
        while gen < need and (len(to_add) + X.shape[0]) < cap_total:
            attempts += 1
            src_idx = int(np.random.choice(idxs))
            base = X[src_idx]
            if np.random.rand() < 0.6:
                aug = augment_jitter(base)
            else:
                aug = augment_time_shift(base, max_shift=2)
                aug = augment_jitter(aug, sigma_min=0.0003, sigma_max=0.0015)
            to_add.append(aug); to_add_labels.append(int(cls)); gen += 1
            if attempts > need * 10: break
    if to_add:
        X_new = np.concatenate([X, np.stack(to_add, axis=0)], axis=0)
        y_new = np.concatenate([y, np.array(to_add_labels, dtype=np.int64)], axis=0)
        return X_new, y_new
    return X, y

def _drop_duplicate_windows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X is None or len(X) == 0: return X, np.arange(len(X))
    seen = {}; keep_idx = []
    for i in range(len(X)):
        h = hashlib.sha256(X[i].astype(np.float32).tobytes()).hexdigest()[:16]
        if h in seen: continue
        seen[h] = i; keep_idx.append(i)
    return X[keep_idx], np.array(keep_idx, dtype=np.int64)

# ========================= 데이터셋 생성(라벨→서명수익→diff) =========================

def create_dataset(features, window=10, strategy="단기", input_size=None):
    """피처 리스트 → 스케일 → 윈도우 → 라벨. close/high/low 유효성과 최신 구간 길이 검증 강화."""
    import pandas as _pd

    def _dummy(symbol_name):
        X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        y = np.zeros((1,), dtype=np.int64)
        return X, y

    symbol_name = "UNKNOWN"
    if isinstance(features, list) and features and isinstance(features[0], dict) and "symbol" in features[0]:
        symbol_name = str(features[0]["symbol"]).upper()
    if not isinstance(features, list) or len(features) <= window:
        safe_failed_result(symbol_name, strategy, reason="not_enough_rows<window")
        return _dummy(symbol_name)

    try:
        df = _pd.DataFrame(features)
        df["timestamp"] = _parse_ts_series(df.get("timestamp"))
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.drop(columns=["strategy"], errors="ignore")

        # 라벨용 컬럼 유효성 체크
        for c in ["close", "high", "low"]:
            df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        df[["close", "high", "low"]] = df[["close", "high", "low"]].ffill()
        df = df.dropna(subset=["close", "high", "low"])

        # 스케일링 대상 숫자화
        feature_cols = [c for c in df.columns if c != "timestamp"]
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=feature_cols)

        if len(df) <= window:
            safe_failed_result(symbol_name, strategy, reason="after_clean_rows<window")
            return _dummy(symbol_name)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols].astype(np.float32))
        df_s = _pd.DataFrame(scaled.astype(np.float32), columns=feature_cols)
        df_s["timestamp"] = df["timestamp"].values
        input_cols = [c for c in df_s.columns if c != "timestamp"]

        # 입력 차원 정합
        target_input = input_size if input_size else max(MIN_FEATURES, len(input_cols))
        if len(input_cols) < target_input:
            for i in range(len(input_cols), target_input):
                padc = f"pad_{i}"
                df_s[padc] = np.float32(0.0)
                input_cols.append(padc)
        elif len(input_cols) > target_input:
            keep = input_cols[:target_input]
            df_s = df_s[keep + ["timestamp"]]
            input_cols = keep

        # 정식 라벨 시도
        y_seq = None
        class_ranges_used = None
        if _make_labels is not None:
            _, labels_full, class_ranges = _make_labels(
                df[["timestamp", "close", "high", "low"]].rename(columns={"timestamp": "timestamp"}),
                symbol=symbol_name,
                strategy=strategy,
                group_id=None,
            )
            if labels_full is not None and len(labels_full) == len(df):
                y_seq = labels_full[window:len(df)]
                class_ranges_used = class_ranges

        # 윈도우 생성
        samples = []
        for i in range(window, len(df_s)):
            seq = df_s.iloc[i - window : i]
            if len(seq) != window:
                continue
            samples.append([[float(seq.iloc[j].get(c, 0.0)) for c in input_cols] for j in range(window)])

        if samples and y_seq is not None:
            X = np.array(samples, dtype=np.float32)
            y = np.array(y_seq[: len(X)], dtype=np.int64)
            keep = np.where(y >= 0)[0]
            if keep.size == 0:
                return _dummy(symbol_name)
            X, y = X[keep], y[keep]
        else:
            # 백업 라벨: 서명수익 → 없으면 diff
            raw = df.to_dict(orient="records")
            lookahead = {"단기": 240, "중기": 1440, "장기": 10080}.get(strategy, 1440)
            signed_vals, samples = [], []
            for i in range(window, len(df_s)):
                seq = df_s.iloc[i - window : i]
                base = raw[i]
                t0 = pd.to_datetime(base.get("timestamp"), errors="coerce", utc=True).tz_convert("Asia/Seoul")
                p0 = float(base.get("close", 0.0))
                if pd.isnull(t0) or p0 <= 0:
                    continue
                fut = [r for r in raw[i + 1 :] if (pd.to_datetime(r.get("timestamp"), utc=True) - t0) <= _pd.Timedelta(minutes=lookahead)]
                if len(seq) != window or not fut:
                    continue
                v_high = [float(r.get("high", r.get("close", p0))) for r in fut if float(r.get("high", r.get("close", p0))) > 0]
                v_low = [float(r.get("low", r.get("close", p0))) for r in fut if float(r.get("low", r.get("close", p0))) > 0]
                if not v_high and not v_low:
                    continue
                up = (max(v_high) - p0) / (p0 + 1e-6) if v_high else 0.0
                dn = (min(v_low) - p0) / (p0 + 1e-6) if v_low else 0.0
                signed_vals.append(float(up if abs(up) >= abs(dn) else dn))
                samples.append([[float(seq.iloc[j].get(c, 0.0)) for c in input_cols] for j in range(window)])

            if samples and signed_vals:
                ranges = cfg_get_class_ranges(symbol=symbol_name, strategy=strategy)
                class_ranges_used = ranges
                y = _label_with_edges(np.asarray(signed_vals, dtype=np.float64), ranges)
                X = np.array(samples, dtype=np.float32)
            else:
                closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float32)
                if len(closes) <= window + 1:
                    return _dummy(symbol_name)
                pct = np.diff(closes) / (closes[:-1] + 1e-6)
                fb_samples, fb_vals = [], []
                for i in range(window, len(df) - 1):
                    seq_rows = df_s.iloc[i - window : i]
                    fb_samples.append([[float(seq_rows.iloc[j].get(c, 0.0)) for c in input_cols] for j in range(window)])
                    fb_vals.append(float(pct[i] if i < len(pct) else 0.0))
                if not fb_samples:
                    return _dummy(symbol_name)
                ranges = cfg_get_class_ranges(symbol=symbol_name, strategy=strategy)
                class_ranges_used = ranges
                y = _label_with_edges(np.asarray(fb_vals, dtype=np.float64), ranges)
                X = np.array(fb_samples, dtype=np.float32)

        # 길이 일치
        m = min(len(X), len(y))
        if m == 0:
            return _dummy(symbol_name)
        X, y = X[:m], y[:m]

        # 중복 제거
        X_dedup, keep_idx = _drop_duplicate_windows(X)
        if len(keep_idx) < len(y):
            y = y[keep_idx]
            X = X_dedup

        # === CHANGE === 1) 클래스 다양성 폴백(uniq<3 → 퍼센타일 리라벨)
        try:
            uniq = np.unique(y)
            if uniq.size < 3:
                # 가능한 값 시퀀스 확보
                vals = None
                if "signed_vals" in locals() and len(signed_vals) >= len(y):
                    vals = np.asarray(signed_vals[: len(y)], dtype=np.float64)
                else:
                    closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
                    if len(closes) > window + 1:
                        pct = np.diff(closes) / (closes[:-1] + 1e-6)
                        vals = np.asarray(pct[-len(y):], dtype=np.float64)
                if vals is not None and vals.size == len(y):
                    n_bins = int(cfg_get_NUM_CLASSES()) or 8
                    qs = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
                    # 경계쌍으로 변환
                    relabel_ranges = [(float(qs[i]), float(qs[i+1])) for i in range(n_bins)]
                    y_new = _label_with_edges(vals, relabel_ranges)
                    if np.unique(y_new).size >= 3:
                        y = y_new
                        class_ranges_used = relabel_ranges
        except Exception:
            pass

        # === CHANGE === 2) 경계 근접 보강 — BOUNDARY_BAND 연동 + env 오버라이드
        try:
            eps_bp_env = os.getenv("BOUNDARY_EPS_BP", None)
            if eps_bp_env is not None:
                eps = max(0.0, float(eps_bp_env) / 10000.0)
            else:
                eps = float(BOUNDARY_BAND)  # config.py의 BOUNDARY_BAND 사용
            if class_ranges_used is not None and len(class_ranges_used) > 1 and eps > 0:
                stops = np.array([b for (_, b) in class_ranges_used[:-1]], dtype=np.float64)
                vals = None
                if "signed_vals" in locals() and len(signed_vals) >= len(y):
                    vals = np.asarray(signed_vals[: len(y)], dtype=np.float64)
                else:
                    closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
                    pct = np.diff(closes) / (closes[:-1] + 1e-6)
                    vals = np.asarray(pct[-len(y):], dtype=np.float64)
                near = np.any(np.abs(vals[:, None] - stops[None, :]) <= eps, axis=1)
                idx_edge = np.where(near)[0]
                if idx_edge.size > 0:
                    dup = min(len(idx_edge), max(1, len(y) // 20))
                    X = np.concatenate([X, X[idx_edge[:dup]]], axis=0)
                    y = np.concatenate([y, y[idx_edge[:dup]]], axis=0)
        except Exception:
            pass

        # === CHANGE === 3) 클래스 최소 샘플 보장(CV min_per_class) — 증강 기반 보완
        try:
            cv_cfg = get_CV_CONFIG()
            min_per_class = int(cv_cfg.get("min_per_class", 3))
            if min_per_class > 0:
                uniq, cnts = np.unique(y, return_counts=True)
                if uniq.size > 0 and np.any(cnts < min_per_class):
                    X, y = augment_for_min_count(X, y, target_count=min_per_class)
        except Exception:
            pass

        # 추가적으로 심각한 쏠림이면 상한까지 보강(기존 로직 유지)
        try:
            if int(os.getenv("AUG_ENABLE", "1")) == 1 and len(y) > 0:
                uniq, cnts = np.unique(y, return_counts=True)
                if cnts.size > 0 and (np.max(cnts) / float(len(y))) >= 0.5:
                    X, y = augment_for_min_count(X, y, target_count=int(np.max(cnts)))
        except Exception:
            pass

        # 메타
        class_ranges_final = class_ranges_used or cfg_get_class_ranges(symbol=symbol_name, strategy=strategy)
        X.attrs = {
            "class_ranges": class_ranges_final,
            "class_groups": cfg_get_class_groups(len(class_ranges_final), 5),
        }
        return X, y

    except Exception:
        safe_failed_result(symbol_name, strategy, reason="create_dataset 예외")
        return _dummy(symbol_name)
# ========================= 추론/데이터셋 헬퍼 =========================
def _select_feature_columns(feat_df: pd.DataFrame) -> List[str]:
    if feat_df is None or feat_df.empty: return []
    cols = [c for c in feat_df.columns if c != "timestamp"]
    # 시간 순으로 의미 있는 피처 우선
    pri = ["open","high","low","close","volume","rsi","macd","macd_signal","macd_hist","ema20","ema50","ema100","ema200",
           "bb_width","bb_percent_b","atr","stoch_k","stoch_d","williams_r","volatility","roc","vwap","trend_score"]
    ordered = [c for c in pri if c in cols]
    rest = [c for c in cols if c not in ordered]
    return ordered + sorted(rest)

def get_feature_window_for_inference(symbol: str, strategy: str, window: int, input_size: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    df_price = get_kline_by_strategy(symbol, strategy)
    feat = compute_features(symbol, df_price, strategy)
    meta = {"recent_rows": int(getattr(feat, "attrs", {}).get("recent_rows", len(df_price))) if isinstance(feat, pd.DataFrame) else 0}
    if not isinstance(feat, pd.DataFrame) or feat.empty or len(feat) < window:
        meta["not_enough_rows"] = True
        X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        return X, meta
    cols = _select_feature_columns(feat)
    mat = feat[cols].tail(window).to_numpy(dtype=np.float32)
    # 패딩 또는 자르기
    F = input_size if input_size else max(MIN_FEATURES, mat.shape[1])
    if mat.shape[1] < F:
        pad = np.zeros((mat.shape[0], F - mat.shape[1]), dtype=np.float32)
        mat = np.concatenate([mat, pad], axis=1)
    elif mat.shape[1] > F:
        mat = mat[:, :F]
    X = mat[np.newaxis, :, :].astype(np.float32)
    X.setflags(write=False)
    meta["input_size"] = F; meta["window"] = window
    return X, meta

def get_inference_batch(symbols: List[str], strategy: str, window: int, input_size: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    out = {}
    for s in symbols:
        try:
            X, meta = get_feature_window_for_inference(s, strategy, window, input_size)
            out[s] = (X, meta)
        except Exception as e:
            safe_failed_result(s, strategy, reason=f"infer_window 실패: {e}")
            X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
            out[s] = (X, {"error": str(e), "window": window, "input_size": X.shape[-1]})
    return out

# ========================= 디스크 I/O 유틸 =========================
_IO_DIR = _ensure_dir(os.getenv("FEATURE_DUMP_DIR", "/persistent/feat"), "/tmp")

def dump_features(symbol: str, strategy: str, df_feat: pd.DataFrame) -> Optional[str]:
    try:
        if df_feat is None or df_feat.empty: return None
        ts_max = _parse_ts_series(df_feat["timestamp"]).max()
        fn = f"{symbol}_{strategy}_{int(pd.Timestamp(ts_max).timestamp())}.parquet"
        path = os.path.join(_IO_DIR, fn)
        df_feat.to_parquet(path, index=False)
        return path
    except Exception as e:
        print(f"[⚠️ dump_features 실패] {e}")
        return None

def load_latest_features(symbol: str, strategy: str) -> Optional[pd.DataFrame]:
    try:
        pats = glob.glob(os.path.join(_IO_DIR, f"{symbol}_{strategy}_*.parquet"))
        if not pats: return None
        pats.sort(reverse=True)
        return pd.read_parquet(pats[0])
    except Exception:
        return None

# ========================= 공개 API =========================
def build_training_dataset(symbol: str, strategy: str, window: int, input_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    df_price = get_kline_by_strategy(symbol, strategy)
    feat_df = compute_features(symbol, df_price, strategy)
    if not isinstance(feat_df, pd.DataFrame) or feat_df.empty:
        X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        y = np.zeros((1,), dtype=np.int64)
        return X, y, {"error": "no_features"}
    # 모델 학습용 리스트 레코드로 변환
    records = feat_df.copy()
    records["symbol"] = symbol
    recs = records.to_dict(orient="records")
    X, y = create_dataset(recs, window=window, strategy=strategy, input_size=input_size)
    meta = {"n": int(len(y)), "F": int(X.shape[-1])}
    return X, y, meta

def get_price_source(symbol: str, strategy: str) -> str:
    df = get_kline_by_strategy(symbol, strategy)
    return getattr(df, "attrs", {}).get("source_exchange", "UNKNOWN")

def enough_for_training(symbol: str, strategy: str) -> bool:
    df = get_kline_by_strategy(symbol, strategy)
    return bool(getattr(df, "attrs", {}).get("enough_for_training", False))

def not_enough_for_predict(symbol: str, strategy: str) -> bool:
    df = get_kline_by_strategy(symbol, strategy)
    return bool(getattr(df, "attrs", {}).get("not_enough_rows", False))

# ========================= 정합성 체크 =========================
def _self_check(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    out = {}
    for strat in ("단기","중기","장기"):
        try:
            dfp = get_kline_by_strategy(symbol, strat)
            feat = compute_features(symbol, dfp, strat)
            ok = isinstance(feat, pd.DataFrame) and not feat.empty
            X, meta = get_feature_window_for_inference(symbol, strat, window=max(10, _PREDICT_MIN_WINDOW))
            out[strat] = {
                "price_rows": int(len(dfp)),
                "feat_rows": int(len(feat)) if ok else 0,
                "win_shape": tuple(X.shape),
                "src": getattr(dfp, "attrs", {}).get("source_exchange", "UNKNOWN"),
                "ok": ok and X.shape[1] >= _PREDICT_MIN_WINDOW
            }
        except Exception as e:
            out[strat] = {"error": str(e)}
    return out

# ========================= 내보내기 =========================
__all__ = [
    # 캐시/상태
    "clear_price_cache","CacheManager",
    # 심볼/그룹
    "get_ALL_SYMBOLS","get_SYMBOL_GROUPS","should_train_symbol","mark_symbol_trained","ready_for_group_predict",
    "mark_group_predicted","get_current_group_index","get_current_group_symbols","reset_group_order",
    "rebuild_symbol_groups","group_all_complete",
    # 수집
    "get_kline","get_kline_binance","get_kline_by_strategy","get_merged_kline_by_strategy","get_kline_interval",
    "get_realtime_prices",
    # 피처
    "compute_features","compute_features_multi","future_gains","future_gains_by_hours",
    # 데이터셋
    "create_dataset","augment_jitter","augment_time_shift","augment_for_min_count",
    # 추론 헬퍼
    "get_feature_window_for_inference","get_inference_batch",
    # I/O
    "dump_features","load_latest_features",
    # 기타
    "get_price_source","enough_for_training","not_enough_for_predict","_self_check",
             ]
# === 초기화 문제 해결용 추가 코드 ===

def clear_all_caches():
    """캐시를 전부 비우는 함수 (초기화용)"""
    try:
        from cache import CacheManager
        CacheManager.clear()
    except Exception:
        pass

# 다른 파일들이 _feature_cache 라는 이름을 찾을 때 에러 안 나게 호환용 클래스
class _CompatFeatureCache:
    @staticmethod
    def clear():
        clear_all_caches()

# 예전 코드에서 불러도 작동하도록 이 이름 등록
_feature_cache = _CompatFeatureCache
