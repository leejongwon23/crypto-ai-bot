# data/utils.py — 안정성/정합성 강화판 (MTF 피처화 + 경계보강/버킷균형 + augmented 소수클래스 증강 + 3단계 컨텍스트 호환 + ✅ 온체인 컨텍스트)
# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}

import os
import time
import json
import requests
import pandas as pd
import numpy as np
import pytz
import glob
from sklearn.preprocessing import MinMaxScaler
from requests.exceptions import HTTPError, RequestException
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import hashlib
import random

# 🔗 라벨 경계/그룹은 config에서만 관리 (일원화)
from config import (
    get_class_ranges as cfg_get_class_ranges,
    get_class_groups as cfg_get_class_groups,
    get_NUM_CLASSES as cfg_get_NUM_CLASSES,
    get_EVAL_RUNTIME,  # ✅ 슬랙 기본값 반영용
)

BASE_URL = "https://api.bybit.com"
BINANCE_BASE_URL = "https://fapi.binance.com"
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuantWorker/1.0; +https://example.com/bot)"}
BINANCE_ENABLED = int(os.getenv("ENABLE_BINANCE", "1"))

# 예측용 최소 윈도우(부족 시 not_enough_rows=True로 플래그) — predict.py에서 스킵
_PREDICT_MIN_WINDOW = int(os.getenv("PREDICT_WINDOW", "10"))

# --- (3단계) 추가 모듈: 선택적 import (없으면 자동 무시) ------------------------
try:
    from features.market import get_market_context_df as _get_market_ctx
except Exception:
    def _get_market_ctx(ts: pd.Series, strategy: str, symbol: Optional[str] = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp"])

try:
    from features.correlations import get_rolling_corr_df as _get_corr_ctx
except Exception:
    def _get_corr_ctx(symbol: str, ts: pd.Series, strategy: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp"])

try:
    from features.regime import get_regime_tags_df as _get_ext_regime_ctx
except Exception:
    def _get_ext_regime_ctx(ts: pd.Series, strategy: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp"])

# ✅ 추가: 온체인 컨텍스트 (없으면 안전 무시)
try:
    from features.onchain import get_onchain_context_df as _get_onchain_ctx
except Exception:
    def _get_onchain_ctx(ts: pd.Series, strategy: str, symbol: Optional[str] = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp"])

def _guess_tolerance_by_strategy(strategy: str) -> pd.Timedelta:
    iv = {"단기": pd.Timedelta(hours=2), "중기": pd.Timedelta(hours=12), "장기": pd.Timedelta(hours=12)}
    return iv.get(strategy, pd.Timedelta(hours=1))

def _merge_asof_all(base: pd.DataFrame, add_list: List[pd.DataFrame], strategy: str) -> pd.DataFrame:
    out = base.copy()
    tol = _guess_tolerance_by_strategy(strategy)
    for add in add_list:
        if add is None or add.empty: continue
        add = add.copy()
        if "timestamp" not in add.columns: continue
        add["timestamp"] = _parse_ts_series(add["timestamp"])
        cols = [c for c in add.columns if c != "timestamp"]
        if not cols: continue
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            add[["timestamp"] + cols].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=tol,
        )
    return out

# --- 기본(백업) 심볼 시드: 최후 fallback 용 ---
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
        return {"enabled": False, "atr_window": 14, "rsi_window": 14,
                "trend_window": 50, "vol_high_pct": 0.9, "vol_low_pct": 0.5}
    _FIS = 24

def _map_bybit_interval(interval: str) -> str:
    mapping = {"60": "60", "1h": "60","120": "120","2h": "120",
               "240": "240","4h": "240","360": "360","6h": "360",
               "720": "720","12h": "720","D": "D","1d": "D","W": "W","M": "M"}
    return mapping.get(str(interval), str(interval))

def _bybit_interval_minutes(mapped: str) -> int:
    m = str(mapped)
    if m.isdigit():  return int(m)
    if m == "D":    return 1440
    if m == "W":    return 10080
    if m == "M":    return 43200
    return 60

def _parse_ts_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce", utc=True)
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            ts = s.copy()
            try:
                if getattr(ts.dt, "tz", None) is None:
                    ts = ts.dt.tz_localize("UTC")
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

def _merge_unique(*lists):
    seen, out = set(), []
    for L in lists:
        for x in L:
            if x and x not in seen:
                seen.add(x); out.append(x)
    return out

def _discover_from_env():
    raw = os.getenv("PREDICT_SYMBOLS") or os.getenv("SYMBOLS_OVERRIDE") or ""
    if not raw.strip():
        return []
    return [p.strip().upper() for p in raw.replace("\n", ",").replace(" ", ",").split(",") if p.strip()]

def _discover_from_models():
    md = "/persistent/models"
    if not os.path.isdir(md): return []
    syms = []
    for fn in os.listdir(md):
        if not (fn.endswith(".pt") or fn.endswith(".meta.json") or fn.endswith(".ptz")):
            continue
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

# ✅ [고정화] 심볼/그룹 — 앞 40개만 사용 (5개씩 8그룹)
SYMBOLS = list(_BASELINE_SYMBOLS[:40])
SYMBOL_GROUPS = _compute_groups(SYMBOLS, 5)

SYMBOL_MAP["bybit"]  = {s: s for s in SYMBOLS}
SYMBOL_MAP["binance"] = {s: s for s in SYMBOLS}

def get_ALL_SYMBOLS(): return list(SYMBOLS)
def get_SYMBOL_GROUPS(): return list(SYMBOL_GROUPS)

# ========================= 그룹 순서 제어(지속성) =========================
_STATE_DIR = "/persistent/state"
_STATE_PATH = os.path.join(_STATE_DIR, "group_order.json")
_STATE_BAK  = _STATE_PATH + ".bak"

# 🔒 예측 게이트 상태 확인(읽기 전용)
_RUN_DIR = "/persistent/run"
_PREDICT_GATE = os.path.join(_RUN_DIR, "predict_gate.json")
def _is_predict_gate_open() -> bool:
    try:
        if not os.path.exists(_PREDICT_GATE):
            return False
        with open(_PREDICT_GATE, "r", encoding="utf-8") as f:
            o = json.load(f)
        return bool(o.get("open", False))
    except Exception:
        return False

def _atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        try: f.flush(); os.fsync(f.fileno())
        except Exception: pass
    os.replace(tmp, path)
    try:
        dfd = os.open(os.path.dirname(path), os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except Exception:
        pass

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
                print(f"[🧭 그룹상태 로드] idx={self.idx}, last_predicted_idx={self.last_predicted_idx}, trained_keys={list(self.trained.keys())}")
        except Exception as e:
            print(f"[⚠️ 그룹상태 로드 실패] {e}")

    def _save(self):
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            payload = {
                "groups": self.groups,
                "idx": self.idx,
                "trained": {k: list(v) for k, v in self.trained.items()},
                "symbols": SYMBOLS,
                "last_predicted_idx": self.last_predicted_idx,
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
            if os.getenv("TRAIN_FORCE_IGNORE_SHOULD", "0") == "1":
                return True
            return not _models_exist()
        except Exception:
            return False

    def should_train(self, symbol: str) -> bool:
        if self._force_allow():
            print(f"[order-override(utils)] {symbol}: force allow (cold-start/env)")
            return True
        i = self.current_index()
        gset = set(self.current_group())
        done = self.trained.get(i, set())
        ok = (symbol in gset) and (symbol not in done)
        if not ok:
            where = "다음 그룹" if symbol not in gset else "이미 학습됨"
            print(f"[⛔ 순서강제] {symbol} → 현재 그룹{i} 차례 아님 ({where})")
        try:
            gate = "open" if _is_predict_gate_open() else "closed"
            print(f"[order] group={i} gate={gate}")
        except Exception:
            pass
        return ok

    def mark_symbol_trained(self, symbol: str):
        i = self.current_index()
        self.trained.setdefault(i, set()).add(symbol)
        self._save()
        print(f"[🧩 학습기록] 그룹{i} 진행중: {sorted(list(self.trained[i]))} / {self.current_group()}")

    def ready_for_group_predict(self) -> bool:
        i = self.current_index()
        group = set(self.current_group())
        done = self.trained.get(i, set())
        all_trained = group.issubset(done) and len(group) > 0
        already_pred = (self.last_predicted_idx == i)
        if all_trained and not already_pred:
            print(f"[🚦 예측준비] 그룹{i} 완주({len(done)}/{len(group)}) → 예측 실행 OK")
            return True
        if already_pred:
            print(f"[⏸ 예측보류] 그룹{i}는 이미 예측 처리됨(last_predicted_idx={self.last_predicted_idx})")
        else:
            remaining = sorted(list(group - done))
            print(f"[⏳ 대기] 그룹{i} 미완료 심볼: {remaining} ({len(done)}/{len(group)})")
        return False

    def mark_group_predicted(self):
        i = self.current_index()
        if self.last_predicted_idx == i:
            print(f"[🛡 중복차단] 그룹{i} 예측 완료가 이미 반영됨 → 스킱")
            return
        print(f"[✅ 예측완료] 그룹{i} → 다음 그룹으로 이동")
        self.last_predicted_idx = i
        self.idx = (i + 1) % max(1, len(self.groups))
        self.trained.setdefault(self.idx, set())
        self._save()

    def reset(self, start_index: int = 0):
        self.idx = max(0, min(start_index, max(0, len(self.groups) - 1)))
        self.trained = {self.idx: set()}
        self.last_predicted_idx = -1
        self._save()
        print(f"[♻️ 그룹순서 리셋] idx={self.idx}")

    def rebuild_groups(self, symbols: Optional[List[str]] = None, group_size: int = 5):
        self.groups = _compute_groups(symbols or SYMBOLS, group_size)[:8]
        self.reset(0)
        print(f"[🧱 그룹재구성] 총 {len(symbols or SYMBOLS)}개 → {len(self.groups)}그룹")

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
    i = GROUP_MGR.current_index()
    group = set(GROUP_MGR.current_group())
    done = GROUP_MGR.trained.get(i, set())
    return (len(group) > 0) and group.issubset(done)

def _models_exist(model_dir="/persistent/models"):
    try:
        if not os.path.isdir(model_dir): return False
        for fn in os.listdir(model_dir):
            full = os.path.join(model_dir, fn)
            if os.path.isdir(full):
                for _, _, files in os.walk(full):
                    if any(f.endswith((".pt", ".ptz", ".meta.json")) for f in files):
                        return True
            else:
                if fn.endswith((".pt", ".ptz", ".meta.json")): return True
        return False
    except Exception:
        return False

class CacheManager:
    _cache = {}
    _ttl = {}
    @classmethod
    def get(cls, key, ttl_sec=None):
        now = time.time()
        if key in cls._cache:
            if ttl_sec is None or now - cls._ttl.get(key, 0) < ttl_sec:
                return cls._cache[key]
            else:
                cls.delete(key)
        return None
    @classmethod
    def set(cls, key, value):
        cls._cache[key] = value
        cls._ttl[key] = time.time()
    @classmethod
    def delete(cls, key):
        if key in cls._cache:
            del cls._cache[key]
            cls._ttl.pop(key, None)
    @classmethod
    def clear(cls):
        cls._cache.clear(); cls._ttl.clear(); print("[캐시 CLEAR ALL]")

def _auto_reset_group_state_if_needed():
    force = os.getenv("FORCE_RESET_GROUPS", "0") == "1"
    no_models = not _models_exist()
    if force or no_models:
        try:
            GROUP_MGR.reset(0)
            try: CacheManager.clear()
            except Exception: pass
            print(f"[♻️ AUTO-RESET] 그룹 상태 자동 리셋 수행 (force={force}, no_models={no_models})")
        except Exception as e:
            print(f"[⚠️ AUTO-RESET 실패] {e}")

_auto_reset_group_state_if_needed()

# ========================= 캐시/백오프 =========================
def _binance_blocked_until(): return CacheManager.get("binance_blocked_until")
def _is_binance_blocked():
    until = _binance_blocked_until()
    return until is not None and time.time() < until
def _block_binance_for(seconds=1800):
    CacheManager.set("binance_blocked_until", time.time() + seconds)
    print(f"[🚫 Binance 차단] {seconds}초 동안 Binance 폴백 비활성화")

def _bybit_blocked_until(): return CacheManager.get("bybit_blocked_until")
def _is_bybit_blocked():
    until = _bybit_blocked_until()
    return until is not None and time.time() < until
def _block_bybit_for(seconds=900):
    CacheManager.set("bybit_blocked_until", time.time() + seconds)
    print(f"[🚫 Bybit 차단] {seconds}초 동안 Bybit 1차 수집 비활성화")

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
    H = pd.Timedelta(hours=int(horizon_hours))
    j0 = 0
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
                if prefer_float32 and df[c].dtype == np.float64:
                    df[c] = df[c].astype(np.float32)
        except Exception:
            pass
    return df

def _fix_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    df[["open","high","low","close"]] = df[["open","high","low","close"]].replace([np.inf, -np.inf], np.nan)
    df["volume"] = df["volume"].replace([np.inf, -np.inf], np.nan)
    df.loc[df[["open","high","low","close"]].le(0).any(axis=1), ["open","high","low","close"]] = np.nan
    df.loc[df["volume"] < 0, "volume"] = np.nan
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].ffill()
    df = df.dropna(subset=["open","high","low","close","volume"])
    mx = df[["open","close","low"]].max(axis=1)
    mn = df[["open","close","high"]].min(axis=1)
    df["high"] = np.maximum(df["high"].values, mx.values)
    df["low"]  = np.minimum(df["low"].values,  mn.values)
    return df

def _winsorize_prices(df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    cols = ["open","high","low","close"]
    for c in cols:
        if c not in df.columns: continue
        s = pd.to_numeric(df[c], errors="coerce")
        lo = s.quantile(lower_q); hi = s.quantile(upper_q)
        df[c] = s.clip(lower=lo, upper=hi).astype(np.float32)
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce")
        lo = max(0.0, v.quantile(lower_q)); hi = v.quantile(upper_q)
        df["volume"] = v.clip(lower=lo, upper=hi).astype(np.float32)
    return df

# ========================= 라벨링 유틸 (경계 기반) =========================
def _label_with_edges(values: np.ndarray, edges: List[Tuple[float, float]]) -> np.ndarray:
    if len(edges) <= 1:
        return np.zeros(len(values), dtype=np.int64)
    stops = np.array([b for (_, b) in edges[:-1]], dtype=np.float64)
    idx = np.digitize(values, stops, right=True)
    idx = np.clip(idx, 0, len(edges)-1)
    return idx.astype(np.int64)

# ========================= 거래소/수집 관련 유틸 =========================
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]
    if df is None:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    ts_candidate = None
    for k in ["timestamp", "start", "t", "open_time", "openTime", "time", 0]:
        if (isinstance(k, int) and k in getattr(df, "columns", [])) or (isinstance(k, str) and k in df.columns):
            ts_candidate = k; break
    if ts_candidate is not None:
        df["timestamp"] = _parse_ts_series(df[ts_candidate])
    else:
        df["timestamp"] = _parse_ts_series(pd.Series([pd.NaT] * len(df)))

    rename_map = {}
    if "1" in df.columns: rename_map["1"] = "open"
    if "2" in df.columns: rename_map["2"] = "high"
    if "3" in df.columns: rename_map["3"] = "low"
    if "4" in df.columns: rename_map["4"] = "close"
    if "5" in df.columns: rename_map["5"] = "volume"
    if rename_map: df = df.rename(columns=rename_map)

    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        else: df[c] = np.nan

    df = df.dropna(subset=["timestamp","open","high","low","close","volume"])
    df["datetime"] = df["timestamp"]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df = _downcast_numeric(df)
    df = _fix_ohlc_consistency(df)
    df = _winsorize_prices(df, 0.001, 0.999)

    try:
        df["timestamp"] = _parse_ts_series(df["timestamp"])
    except Exception:
        pass

    df = df.dropna(subset=["timestamp","open","high","low","close","volume"]).reset_index(drop=True)
    return df[cols]

def _clip_tail(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df is None or df.empty: return df
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    except Exception:
        pass
    mask = ts.diff().fillna(pd.Timedelta(seconds=0)) >= pd.Timedelta(seconds=0)
    if not mask.all():
        df = df[mask].reset_index(drop=True)
    return df

# ========================= 거래소 수집기(Bybit) =========================
def get_kline(symbol: str, interval: str = "60", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    if _is_bybit_blocked():
        print("[⛔ Bybit 비활성화 상태] 일시차단 중")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

    real_symbol = SYMBOL_MAP["bybit"].get(symbol, symbol)
    target_rows = int(limit)
    collected, total, last_oldest = [], 0, None

    interval = _map_bybit_interval(interval)
    iv_minutes = _bybit_interval_minutes(interval)

    start_ms = None
    if end_time is None:
        lookback_ms = int(target_rows * iv_minutes * 60 * 1000)
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        start_ms = max(0, now_ms - lookback_ms)

    empty_resp_count = 0

    while total < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total
                req = min(1000, rows_needed)
                for category in ("linear", "spot"):
                    params = {"category": category, "symbol": real_symbol, "interval": interval, "limit": req}
                    if end_time is not None:
                        params["end"] = int(end_time.timestamp() * 1000)
                    elif start_ms is not None:
                        params["start"] = start_ms
                    res = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10, headers=REQUEST_HEADERS)
                    res.raise_for_status()
                    data = res.json()
                    raw = (data or {}).get("result", {}).get("list", [])
                    if not raw:
                        empty_resp_count += 1
                        continue
                    if isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6:
                        df_chunk = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(raw[0])])
                    else:
                        df_chunk = pd.DataFrame(raw)
                    df_chunk = _normalize_df(df_chunk)
                    if df_chunk.empty:
                        continue
                    collected.append(df_chunk)
                    total += len(df_chunk)
                    success = True
                    oldest_ts = df_chunk["timestamp"].min()
                    if last_oldest is not None and pd.to_datetime(oldest_ts) >= pd.to_datetime(last_oldest):
                        oldest_ts = pd.to_datetime(oldest_ts) - pd.Timedelta(minutes=1)
                    last_oldest = oldest_ts
                    end_time = pd.to_datetime(oldest_ts).tz_convert("UTC") - pd.Timedelta(milliseconds=1)
                    time.sleep(0.2)
                    break
                if success:
                    break
            except RequestException as e:
                time.sleep(1); continue
            except Exception:
                time.sleep(0.5); continue
        if not success:
            break

    if collected:
        df = _normalize_df(pd.concat(collected, ignore_index=True))
        df.attrs["source_exchange"] = "BYBIT"
        return df

    if empty_resp_count >= max_retry * 2:
        _block_bybit_for(int(os.getenv("BYBIT_BACKOFF_SEC", "900")))
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 거래소 수집기(Binance) =========================
def get_kline_binance(symbol: str, interval: str = "240", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    real_symbol = SYMBOL_MAP["binance"].get(symbol, symbol)
    _bin_iv = None
    for _, cfg in STRATEGY_CONFIG.items():
        if cfg.get("interval") == interval:
            _bin_iv = cfg.get("binance_interval"); break
    if _bin_iv is None:
        _bin_iv = {"240": "4h", "D": "1d", "2D": "2d", "60": "1h"}.get(interval, "1h")

    target_rows = int(limit)
    collected, total, last_oldest = [], 0, None

    if not BINANCE_ENABLED or _is_binance_blocked():
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

    while total < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total
                req = min(1000, rows_needed)
                params = {"symbol": real_symbol, "interval": _bin_iv, "limit": req}
                if end_time is not None:
                    params["endTime"] = int(end_time.timestamp() * 1000)
                res = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/klines", params=params, timeout=10, headers=REQUEST_HEADERS)
                try:
                    res.raise_for_status()
                except HTTPError as he:
                    if getattr(he.response, "status_code", None) == 418:
                        _block_binance_for(1800)
                        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])
                    raise
                raw = res.json()
                if not raw:
                    break
                if isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6:
                    df_chunk = pd.DataFrame(raw, columns=[
                        "timestamp","open","high","low","close","volume",
                        "close_time","quote_asset_volume","trades","taker_base_vol","taker_quote_vol","ignore"
                    ])
                else:
                    df_chunk = pd.DataFrame(raw)
                df_chunk = _normalize_df(df_chunk)
                if df_chunk.empty:
                    break
                collected.append(df_chunk)
                total += len(df_chunk)
                success = True
                if total >= target_rows:
                    break
                oldest_ts = df_chunk["timestamp"].min()
                if last_oldest is not None and pd.to_datetime(oldest_ts) >= pd.to_datetime(last_oldest):
                    oldest_ts = pd.to_datetime(oldest_ts) - pd.Timedelta(minutes=1)
                last_oldest = oldest_ts
                end_time = pd.to_datetime(oldest_ts).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)
                time.sleep(0.3)
                break
            except RequestException:
                time.sleep(1); continue
            except Exception:
                time.sleep(0.5); continue
        if not success:
            break

    if collected:
        df = _normalize_df(pd.concat(collected, ignore_index=True))
        df.attrs["source_exchange"] = "BINANCE"
        return df
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 통합 수집 + 병합 =========================
def get_merged_kline_by_strategy(symbol: str, strategy: str) -> pd.DataFrame:
    config = STRATEGY_CONFIG.get(strategy)
    if not config:
        return pd.DataFrame()
    interval = config["interval"]; base_limit = int(config["limit"]); max_total = base_limit

    def fetch_until_target(fetch_func, src):
        total = []; end = None; cnt = 0; max_rep = 10
        while cnt < max_total and len(total) < max_rep:
            dfc = fetch_func(symbol, interval=interval, limit=base_limit, end_time=end)
            if dfc is None or dfc.empty:
                break
            total.append(dfc); cnt += len(dfc)
            if len(dfc) < base_limit:
                break
            oldest = dfc["timestamp"].min()
            end = pd.to_datetime(oldest).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)
        dff = _normalize_df(pd.concat(total, ignore_index=True)) if total else pd.DataFrame()
        return dff

    df_bybit = fetch_until_target(get_kline, "Bybit")
    df_binance = pd.DataFrame()
    if len(df_bybit) < base_limit and BINANCE_ENABLED and not _is_binance_blocked():
        df_binance = fetch_until_target(get_kline_binance, "Binance")

    df_all = _normalize_df(pd.concat([df_bybit, df_binance], ignore_index=True)) if (not df_bybit.empty or not df_binance.empty) else pd.DataFrame()
    if df_all.empty:
        return pd.DataFrame()

    df_all = _clip_tail(df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), base_limit)

    srcs = []
    if not df_bybit.empty: srcs.append("BYBIT")
    if not df_binance.empty: srcs.append("BINANCE")
    df_all.attrs["source_exchange"] = "+".join(srcs) if srcs else "UNKNOWN"

    for c in ["timestamp","open","high","low","close","volume"]:
        if c not in df_all.columns:
            df_all[c] = 0.0 if c != "timestamp" else pd.Timestamp.now(tz="Asia/Seoul")
    df_all.attrs["augment_needed"] = len(df_all) < base_limit
    return df_all

# =============== 임의 인터벌 전용 수집기(도우미, MTF용) ===============
def get_kline_interval(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    interval 예: '60','240','720','D','W'
    """
    try:
        df_bybit = get_kline(symbol, interval=interval, limit=limit)
        if (df_bybit is None or df_bybit.empty) and not _is_binance_blocked():
            # 단순 폴백(가능한 근접 주기)
            df_bin = get_kline_binance(symbol, interval=interval, limit=limit)
            return _normalize_df(df_bin)
        return _normalize_df(df_bybit)
    except Exception:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 전략별 Kline (캐시 포함) =========================
def get_kline_by_strategy(symbol: str, strategy: str, end_slack_min: int = 0):
    if not end_slack_min:
        try:
            end_slack_min = int(get_EVAL_RUNTIME().get("price_window_slack_min", 10))
        except Exception:
            end_slack_min = 0

    cache_key = f"{symbol}-{strategy}-slack{end_slack_min}"
    cached = CacheManager.get(cache_key, ttl_sec=600)
    if cached is not None:
        return cached
    try:
        cfg = STRATEGY_CONFIG.get(strategy, {"limit": 300, "interval": "D"})
        limit = int(cfg.get("limit", 300)); interval = cfg.get("interval", "D")

        df_bybit = []; total_bybit = 0; end = None
        while total_bybit < limit:
            dfc = get_kline(symbol, interval=interval, limit=limit, end_time=end)
            if dfc is None or dfc.empty:
                break
            df_bybit.append(dfc); total_bybit += len(dfc)
            oldest = dfc["timestamp"].min()
            oldest_utc = pd.to_datetime(oldest).tz_convert("UTC")
            if end_slack_min:
                oldest_utc = oldest_utc - pd.Timedelta(minutes=int(end_slack_min))
            end = oldest_utc - pd.Timedelta(milliseconds=1)
            if len(dfc) < limit:
                break
        df_bybit = _normalize_df(pd.concat(df_bybit, ignore_index=True)) if df_bybit else pd.DataFrame()

        df_binance = []; total_binance = 0
        if len(df_bybit) < int(limit * 0.9) and BINANCE_ENABLED and not _is_binance_blocked():
            end = None
            while total_binance < limit:
                dfc = get_kline_binance(symbol, interval=interval, limit=limit, end_time=end)
                if dfc is None or dfc.empty:
                    break
                df_binance.append(dfc); total_binance += len(dfc)
                oldest = dfc["timestamp"].min()
                oldest_utc = pd.to_datetime(oldest).tz_convert("UTC")
                if end_slack_min:
                    oldest_utc = oldest_utc - pd.Timedelta(minutes=int(end_slack_min))
                end = oldest_utc - pd.Timedelta(milliseconds=1)
                if len(dfc) < limit:
                    break
        df_binance = _normalize_df(pd.concat(df_binance, ignore_index=True)) if df_binance else pd.DataFrame()

        df_list = [d for d in [df_bybit, df_binance] if d is not None and not d.empty]
        df = _normalize_df(pd.concat(df_list, ignore_index=True)) if df_list else pd.DataFrame()
        df = _clip_tail(df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), limit)

        srcs = []
        if not df_bybit.empty: srcs.append("BYBIT")
        if not df_binance.empty: srcs.append("BINANCE")
        df.attrs["source_exchange"] = "+".join(srcs) if srcs else "UNKNOWN"

        total = len(df); min_required = max(60, int(limit * 0.90))
        if total < min_required:
            df_retry = get_merged_kline_by_strategy(symbol, strategy)
            if not df_retry.empty and len(df_retry) > total:
                df = _clip_tail(df_retry, limit); total = len(df)

        df.attrs["augment_needed"] = total < limit
        df.attrs["enough_for_training"] = total >= min_required

        # ▲ 예측용 보조 플래그: 최근 캔들 수/부족 여부 저장
        df.attrs["recent_rows"] = int(total)
        df.attrs["not_enough_rows"] = bool(total < _PREDICT_MIN_WINDOW)

        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("UTC")
            df["datetime"] = df["timestamp"].dt.tz_convert("Asia/Seoul")
        except Exception:
            pass

        CacheManager.set(cache_key, df)
        return df
    except Exception as e:
        safe_failed_result(symbol, strategy, reason=str(e))
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 프리패치 =========================
def prefetch_symbol_groups(strategy: str):
    for group in SYMBOL_GROUPS:
        for sym in group:
            try:
                get_kline_by_strategy(sym, strategy)
            except Exception:
                pass

# ========================= 실시간 티커 =========================
def get_realtime_prices():
    url = f"{BASE_URL}/v5/market/tickers"; params = {"category": "linear"}
    try:
        res = requests.get(url, params=params, timeout=10, headers=REQUEST_HEADERS)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return {}
        tickers = data["result"]["list"]
        symset = set(get_ALL_SYMBOLS())
        return {item["symbol"]: float(item["lastPrice"]) for item in tickers if item["symbol"] in symset}
    except Exception:
        return {}

# ========================= 피처 생성 (MTF + 레짐 + 3단계 컨텍스트) =========================
_feature_cache = {}
def _ensure_columns(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0

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
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_window=14, d_window=3):
    lowest = low.rolling(k_window, min_periods=1).min()
    highest = high.rolling(k_window, min_periods=1).max()
    k = 100 * (close - lowest) / ((highest - lowest) + 1e-6)
    d = k.rolling(d_window, min_periods=1).mean()
    return k, d

def _prefix_cols(df: pd.DataFrame, prefix: str, skip=("timestamp",)) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    ren = {c: f"{prefix}_{c}" for c in df.columns if c not in skip}
    df = df.rename(columns=ren)
    return df

def _compute_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_cols = ["open","high","low","close","volume"]
    for c in base_cols:
        if c not in df.columns: df[c] = 0.0

    df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))

    macd, macd_sig, macd_hist = _macd_parts(df["close"])
    df["macd"] = macd; df["macd_signal"] = macd_sig; df["macd_hist"] = macd_hist

    ma20, bb_up, bb_dn, bb_sd, bb_width, bb_pb = _bbands(df["close"], window=20)
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
    """
    반환: (base_iv, aux_list)
    """
    plan = {
        "단기": ("60",  ["240", "D"]),   # 1h base + 4h, 1d
        "중기": ("240", ["D", "3D"]),   # 4h base + 1d, 3d(합성)
        "장기": ("D",   ["W", "720"]),  # 1d base + 1w, 12h(보조)
    }
    return plan.get(strategy, ("240", ["D"]))

def _synthesize_multi(df_base: pd.DataFrame, target_iv: str) -> pd.DataFrame:
    """
    지원하지 않는 주기는 베이스에서 resample로 합성
    """
    if df_base is None or df_base.empty: return pd.DataFrame()
    df = df_base.copy()
    df["timestamp"] = _parse_ts_series(df["timestamp"]).dt.tz_convert("UTC")
    df = df.set_index("timestamp")

    rule_map = {"3D": "3D", "2D": "2D"}
    if target_iv in rule_map:
        rule = rule_map[target_iv]
    else:
        return pd.DataFrame()

    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum(min_count=1)
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna(how="any").reset_index()
    out["timestamp"] = out["timestamp"].dt.tz_convert("Asia/Seoul")
    out["datetime"] = out["timestamp"]
    return _normalize_df(out)

def _compute_mtf_features(symbol: str, strategy: str, df_base: pd.DataFrame) -> pd.DataFrame:
    base_iv, aux_ivs = _mtf_plan(strategy)

    # 베이스 수집(요청된 df가 해당 인터벌이 아니면 새로 수집)
    df_b = df_base.copy()
    # 보조 타임프레임 수집/합성
    ctx_blocks = []
    for iv in aux_ivs:
        if iv in ("3D","2D"):
            d = _synthesize_multi(df_b, iv)
        else:
            d = get_kline_interval(symbol, iv, limit=max(200, len(df_b)))
        if d is None or d.empty: continue
        feat = _compute_feature_block(d)
        feat = feat[["timestamp","close","rsi","macd","macd_signal","bb_width","atr","ema20","ema50","ema100","ema200","stoch_k","stoch_d","williams_r","roc","trend_score","vwap"]]
        ctx_blocks.append(_prefix_cols(feat, f"f{iv}"))

    # 베이스 피처
    base_feat = _compute_feature_block(df_b)
    base_feat = base_feat[["timestamp","open","high","low","close","volume","rsi","macd","macd_signal","macd_hist","bb_up","bb_dn","bb_sd","bb_width","bb_percent_b","volatility","ema20","ema50","ema100","ema200","trend_score","roc","atr","stoch_k","stoch_d","williams_r","vwap"]]

    # asof 병합
    merged = _merge_asof_all(base_feat, ctx_blocks, strategy)
    return merged

def compute_features(symbol: str, df: pd.DataFrame, strategy: str, required_features: list = None, fallback_input_size: int = None) -> pd.DataFrame:
    cache_key = f"{symbol}-{strategy}-features"
    cached = CacheManager.get(cache_key, ttl_sec=600)
    if cached is not None:
        return cached

    # 입력 점검
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        safe_failed_result(symbol, strategy, reason="입력DataFrame empty")
        dummy = pd.DataFrame()
        dummy.attrs["not_enough_rows"] = True
        return dummy

    # 윈도우 부족 시 바로 플래그 반환 (predict.py가 스킵)
    if len(df) < _PREDICT_MIN_WINDOW:
        dummy = pd.DataFrame()
        dummy.attrs["not_enough_rows"] = True
        dummy.attrs["recent_rows"] = int(len(df))
        return dummy

    df = df.copy()
    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now", utc=True).tz_convert("Asia/Seoul")
    df["strategy"] = strategy

    base_cols = ["open","high","low","close","volume"]
    for c in base_cols:
        if c not in df.columns: df[c] = 0.0
    df = df[["timestamp"] + base_cols]

    if len(df) < 20:
        safe_failed_result(symbol, strategy, reason=f"row 부족 {len(df)}")
        dummy = pd.DataFrame()
        dummy.attrs["not_enough_rows"] = True
        dummy.attrs["recent_rows"] = int(len(df))
        return dummy

    try:
        # ============ MTF 피처화 ============ #
        feat = _compute_mtf_features(symbol, strategy, df)

        # (내부 레짐)
        regime_cfg = get_REGIME()
        if regime_cfg.get("enabled", False):
            try:
                atr_win = int(regime_cfg.get("atr_window", 14))
                trend_win = int(regime_cfg.get("trend_window", 50))
                vol_high = float(regime_cfg.get("vol_high_pct", 0.9))
                vol_low  = float(regime_cfg.get("vol_low_pct", 0.5))
                feat["atr_val"] = _atr(feat["high"], feat["low"], feat["close"], window=atr_win)
                thr_high = feat["atr_val"].quantile(vol_high)
                thr_low  = feat["atr_val"].quantile(vol_low)
                feat["vol_regime"] = np.where(feat["atr_val"] >= thr_high, 2, np.where(fet["atr_val"] <= thr_low, 0, 1))
                feat["ma_trend"] = feat["close"].rolling(window=trend_win, min_periods=1).mean()
                slope = feat["ma_trend"].diff()
                feat["trend_regime"] = np.where(slope > 0, 2, np.where(slope < 0, 0, 1))
                feat["regime_tag"] = feat["vol_regime"] * 3 + feat["trend_regime"]
            except Exception:
                pass

        # (3단계) 외부 시장 컨텍스트 병합
        try:
            ts = _parse_ts_series(feat["timestamp"])
            ctx_list = [
                _get_market_ctx(ts, strategy, symbol),
                _get_corr_ctx(symbol, ts, strategy),
                _get_ext_regime_ctx(ts, strategy),
                _get_onchain_ctx(ts, strategy, symbol),  # ✅ 온체인 컨텍스트 병합
            ]
            feat = _merge_asof_all(feat, ctx_list, strategy)
        except Exception:
            pass

        # 필수 컬럼 보정
        must_have = [
            "rsi","macd","macd_signal","macd_hist",
            "ema20","ema50","ema100","ema200",
            "bb_width","bb_percent_b","atr","stoch_k","stoch_d",
            "williams_r","volatility","roc","vwap","trend_score"
        ]
        _ensure_columns(feat, must_have)

        # ✅ NaN/Inf 정규화 (훈련과 동일)
        feat.replace([np.inf, -np.inf], 0, inplace=True)
        feat.fillna(0, inplace=True)

        feat_cols = [c for c in feat.columns if c != "timestamp"]

        if len(feat_cols) < _FIS:
            for i in range(len(feat_cols), _FIS):
                pad = f"pad_{i}"
                feat[pad] = 0.0; feat_cols.append(pad)

        feat[feat_cols] = _downcast_numeric(feat[feat_cols]).astype(np.float32)
        feat[feat_cols] = MinMaxScaler().fit_transform(feat[feat_cols])

    except Exception as e:
        safe_failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        dummy = pd.DataFrame()
        dummy.attrs["not_enough_rows"] = True
        return dummy

    if feat.empty or feat.isnull().values.any():
        safe_failed_result(symbol, strategy, reason="최종 결과 DataFrame 오류")
        dummy = pd.DataFrame()
        dummy.attrs["not_enough_rows"] = True
        return dummy

    CacheManager.set(cache_key, feat)
    return feat

# ========================= 증강 관련 함수 =========================
def augment_jitter(seq: np.ndarray, sigma_min: float = 0.0005, sigma_max: float = 0.002) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.size == 0: return seq.copy()
    sigma = float(np.random.uniform(sigma_min, sigma_max))
    noise = np.random.normal(loc=0.0, scale=sigma, size=seq.shape).astype(np.float32)
    aug = seq * (1.0 + noise)
    return aug.astype(np.float32)

def augment_time_shift(seq: np.ndarray, max_shift: int = 2) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] <= 1 or max_shift <= 0: return seq.copy()
    shift = int(np.random.randint(-max_shift, max_shift + 1))
    if shift == 0: return seq.copy()
    w, f = seq.shape
    if shift > 0:
        pad = np.repeat(seq[0:1, :], shift, axis=0)
        new = np.vstack([pad, seq[:w - shift, :]])
    else:
        s = -shift
        pad = np.repeat(seq[-1:, :], s, axis=0)
        new = np.vstack([seq[s:, :], pad])
    if new.shape[0] != w:
        if new.shape[0] > w: new = new[:w, :]
        else:
            extra = np.repeat(seq[-1:, :], w - new.shape[0], axis=0)
            new = np.vstack([new, extra])
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

# =============== 중복 창 컷(간단 해시) ===============
def _drop_duplicate_windows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X is None or len(X) == 0: return X, np.arange(len(X))
    seen = {}
    keep_idx = []
    for i in range(len(X)):
        h = hashlib.sha256(X[i].astype(np.float32).tobytes()).hexdigest()[:16]
        if h in seen: continue
        seen[h] = i; keep_idx.append(i)
    return X[keep_idx], np.array(keep_idx, dtype=np.int64)

# ========================= 데이터셋 생성 =========================
def create_dataset(features, window=10, strategy="단기", input_size=None):
    import pandas as _pd
    from config import MIN_FEATURES

    def _dummy(symbol_name):
        from config import MIN_FEATURES as _MINF
        safe_failed_result(symbol_name, strategy, reason="create_dataset 입력 feature 부족/실패")
        X = np.zeros((1, window, input_size if input_size else _MINF), dtype=np.float32)
        y = np.zeros((1,), dtype=np.int64)
        return X, y

    symbol_name = "UNKNOWN"
    if isinstance(features, list) and features and isinstance(features[0], dict) and "symbol" in features[0]:
        symbol_name = features[0]["symbol"]

    if not isinstance(features, list) or len(features) <= window:
        safe_failed_result(symbol_name, strategy, reason="not_enough_rows<window")
        return _dummy(symbol_name)

    try:
        df = _pd.DataFrame(features)
        df["timestamp"] = _parse_ts_series(df.get("timestamp"))
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.drop(columns=["strategy"], errors="ignore")

        num_cols = [c for c in df.columns if c != "timestamp"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[num_cols] = _downcast_numeric(df[num_cols])

        feature_cols = [c for c in df.columns if c != "timestamp"]
        if not feature_cols:
            return _dummy(symbol_name)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols].astype(np.float32))
        df_s = _pd.DataFrame(scaled.astype(np.float32), columns=feature_cols)
        df_s["timestamp"] = df["timestamp"].values

        raw_records = df.to_dict(orient="records")
        input_cols = [c for c in df_s.columns if c != "timestamp"]

        target_input = input_size if input_size else max(MIN_FEATURES, len(input_cols))
        if len(input_cols) < target_input:
            for i in range(len(input_cols), target_input):
                padc = f"pad_{i}"
                df_s[padc] = np.float32(0.0); input_cols.append(padc)
        elif len(input_cols) > target_input:
            keep = set(input_cols[:target_input])
            df_s = df_s.drop(columns=[c for c in input_cols if c not in keep], errors="ignore")
            input_cols = [c for c in input_cols if c in keep]

        strategy_minutes = {"단기": 240, "중기": 1440, "장기": 10080}
        lookahead = strategy_minutes.get(strategy, 1440)

        samples, signed_vals = [], []
        for i in range(window, len(df_s)):
            seq = df_s.iloc[i - window:i]
            base_raw = raw_records[i]
            try:
                entry_time = pd.to_datetime(base_raw.get("timestamp"), errors="coerce", utc=True).tz_convert("Asia/Seoul")
            except Exception:
                continue
            entry_price = float(base_raw.get("close", 0.0))
            if pd.isnull(entry_time) or entry_price <= 0: continue
            try:
                fut_raw = [
                    r for r in raw_records[i + 1:]
                    if (pd.to_datetime(r.get("timestamp", None), utc=True) - entry_time) <= _pd.Timedelta(minutes=lookahead)
                ]
            except Exception: continue

            if len(seq) != window or not fut_raw: continue

            v_highs = [float(r.get("high", r.get("close", entry_price))) for r in fut_raw if float(r.get("high", r.get("close", entry_price))) > 0]
            v_lows  = [float(r.get("low",  r.get("close", entry_price)))  for r in fut_raw if float(r.get("low",  r.get("close", entry_price)))  > 0]

            if not v_highs and not v_lows: continue

            max_future = max(v_highs) if v_highs else entry_price
            min_future = min(v_lows)  if v_lows  else entry_price
            ret_up = (max_future - entry_price) / (entry_price + 1e-6)
            ret_dn = (min_future - entry_price) / (entry_price + 1e-6)
            signed_ret = ret_up if abs(ret_up) >= abs(ret_dn) else ret_dn
            signed_vals.append(float(signed_ret))

            sample = [[float(seq.iloc[j].get(c, 0.0)) for c in input_cols] for j in range(window)]
            samples.append(sample)

        if samples and signed_vals:
            ranges = cfg_get_class_ranges(symbol=symbol_name, strategy=strategy)
            y = _label_with_edges(np.asarray(signed_vals, dtype=np.float64), ranges)
            X = np.array(samples, dtype=np.float32)
            if len(X) != len(y):
                m = min(len(X), len(y)); X = X[:m]; y = y[:m]
            X.attrs = {"class_ranges": ranges, "class_groups": cfg_get_class_groups(len(ranges), 5)}

            # --- 중복 창 컷 ---
            X_dedup, keep_idx = _drop_duplicate_windows(X)
            if len(keep_idx) < len(y):
                y = y[keep_idx]; X = X_dedup

            # --- 경계 보강(±ε) 오버샘플 ---
            try:
                eps_bp = int(os.getenv("BOUNDARY_EPS_BP", "30"))  # 30bp = 0.3%
                eps = eps_bp / 10000.0
                stops = np.array([b for (_, b) in ranges[:-1]], dtype=np.float64)
                vals = np.asarray(signed_vals[:len(y)], dtype=np.float64)
                close_to_edge = np.any(np.abs(vals[:, None] - stops[None, :]) <= eps, axis=1)
                idx_edge = np.where(close_to_edge)[0]
                if idx_edge.size > 0:
                    dup = min(len(idx_edge), max(1, len(y)//20))  # 전체의 ~5% 이내
                    X = np.concatenate([X, X[idx_edge[:dup]]], axis=0)
                    y = np.concatenate([y, y[idx_edge[:dup]]], axis=0)
            except Exception:
                pass

            # --- 레짐×변동성 버킷 균형(옵션) ---
            try:
                if int(os.getenv("BALANCE_BY_BUCKETS", "0")) == 1:
                    vol = pd.to_numeric(df.get("vol_regime", pd.Series([1]*len(df))), errors="coerce").fillna(1).astype(int).values
                    trd = pd.to_numeric(df.get("trend_regime", pd.Series([1]*len(df))), errors="coerce").fillna(1).astype(int).values
                    start = len(vol) - len(y)
                    start = max(0, start)
                    bucket = (vol[start:start+len(y)] * 3 + trd[start:start+len(y)]).astype(int)
                    uniq_b, cnts = np.unique(bucket, return_counts=True)
                    target = int(np.median(cnts)) if len(cnts)>0 else None
                    if target and target>0:
                        sel_idx = []
                        for b in uniq_b:
                            idxs = np.where(bucket == b)[0]
                            if len(idxs) <= target:
                                sel_idx.extend(idxs.tolist())
                            else:
                                sel_idx.extend(np.random.choice(idxs, size=target, replace=False).tolist())
                        sel_idx = np.array(sorted(sel_idx))
                        X = X[sel_idx]; y = y[sel_idx]
            except Exception:
                pass

            # --- 소수클래스 증강 ---
            try:
                if int(os.getenv("AUG_ENABLE","1")) == 1:
                    uniq, cnts = np.unique(y, return_counts=True)
                    if cnts.size > 0:
                        max_cnt = int(np.max(cnts))
                        total = len(y)
                        if total > 0 and (np.max(cnts) / float(total)) >= 0.5:
                            X, y = augment_for_min_count(X, y, target_count=max_cnt)
            except Exception:
                pass

            return X, y

        # Fallback: 단순 수익률
        closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float32)
        if len(closes) <= window + 1: return _dummy(symbol_name)
        pct = np.diff(closes) / (closes[:-1] + 1e-6)
        fb_samples, fb_vals = [], []
        for i in range(window, len(df) - 1):
            seq_rows = df_s.iloc[i - window:i]
            sample = [[float(seq_rows.iloc[j].get(c, 0.0)) for c in input_cols] for j in range(window)]
            fb_samples.append(sample)
            fb_vals.append(float(pct[i] if i < len(pct) else 0.0))
        if not fb_samples: return _dummy(symbol_name)

        ranges = cfg_get_class_ranges(symbol=symbol_name, strategy=strategy)
        y = _label_with_edges(np.asarray(fb_vals, dtype=np.float64), ranges)
        X = np.array(fb_samples, dtype=np.float32)
        if len(X) != len(y):
            m = min(len(X), len(y)); X = X[:m]; y = y[:m]
        X.attrs = {"class_ranges": ranges, "class_groups": cfg_get_class_groups(len(ranges), 5)}

        # 중복컷 + 소수클래스 증강(동일 로직)
        X_dedup, keep_idx = _drop_duplicate_windows(X)
        if len(keep_idx) < len(y):
            y = y[keep_idx]; X = X_dedup
        try:
            if int(os.getenv("AUG_ENABLE","1")) == 1:
                uniq, cnts = np.unique(y, return_counts=True)
                if cnts.size > 0:
                    max_cnt = int(np.max(cnts))
                    total = len(y)
                    if total > 0 and (np.max(cnts) / float(total)) >= 0.5:
                        X, y = augment_for_min_count(X, y, target_count=max_cnt)
        except Exception:
            pass
        return X, y

    except Exception as e:
        return _dummy(symbol_name)
