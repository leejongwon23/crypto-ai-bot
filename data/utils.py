# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}

import os, time, json, requests, pandas as pd, numpy as np, pytz, glob
from sklearn.preprocessing import MinMaxScaler
from requests.exceptions import HTTPError, RequestException
from typing import List, Dict, Any, Optional

BASE_URL = "https://api.bybit.com"
BINANCE_BASE_URL = "https://fapi.binance.com"
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuantWorker/1.0; +https://example.com/bot)"}
BINANCE_ENABLED = int(os.getenv("ENABLE_BINANCE", "1"))

# --- 기본(백업) 심볼 시드 60개: 최후 fallback 용 ---
_BASELINE_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","TRXUSDT",
    "LTCUSDT","BCHUSDT","LINKUSDT","ATOMUSDT","XLMUSDT","ETCUSDT","ICPUSDT","HBARUSDT","FILUSDT","SANDUSDT",
    "RNDRUSDT","INJUSDT","NEARUSDT","APEUSDT","ARBUSDT","AAVEUSDT","OPUSDT","SUIUSDT","DYDXUSDT","CHZUSDT",
    "LDOUSDT","STXUSDT","GMTUSDT","FTMUSDT","WLDUSDT","TIAUSDT","SEIUSDT","ARKMUSDT","JASMYUSDT","AKTUSDT",
    "GMXUSDT","SKLUSDT","BLURUSDT","ENSUSDT","CFXUSDT","FLOWUSDT","ALGOUSDT","MINAUSDT","NEOUSDT","MASKUSDT",
    "KAVAUSDT","BATUSDT","ZILUSDT","WAVESUSDT","OCEANUSDT","1INCHUSDT","YFIUSDT","STGUSDT","GALAUSDT","IMXUSDT"
]
SYMBOL_MAP = {"bybit": {}, "binance": {}}

# 전략 설정/레짐
try:
    from config import STRATEGY_CONFIG, get_REGIME
except Exception:
    STRATEGY_CONFIG = {
        "단기": {"interval": "240", "limit": 1000, "binance_interval": "4h"},
        "중기": {"interval": "D", "limit": 500, "binance_interval": "1d"},
        "장기": {"interval": "D", "limit": 500, "binance_interval": "1d"},
    }
    def get_REGIME():
        return {
            "enabled": False, "atr_window": 14, "rsi_window": 14, "trend_window": 50,
            "vol_high_pct": 0.9, "vol_low_pct": 0.5
        }

# ========================= Bybit interval 매핑 =========================
def _map_bybit_interval(interval: str) -> str:
    """Bybit API가 요구하는 interval 문자열로 변환"""
    mapping = {
        "60": "60", "240": "240", "1h": "60", "4h": "240",
        "D": "D", "1d": "D", "2D": "120", "3D": "180",
        "W": "W", "M": "M"
    }
    return mapping.get(str(interval), str(interval))

# ========================= 공통 유틸 =========================
def _parse_ts_series(s: pd.Series) -> pd.Series:
    """혼합 timestamp(정수 초/밀리초, ISO 문자열, datetime)를 UTC→KST로 안전 변환."""
    if s is None:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce", utc=True)
    try:
        # 이미 datetime이면 tz 보정
        if pd.api.types.is_datetime64_any_dtype(s):
            ts = s.copy()
            try:
                if getattr(ts.dt, "tz", None) is None:
                    ts = ts.dt.tz_localize("UTC")
            except Exception:
                pass
            return ts.dt.tz_convert("Asia/Seoul")

        # 숫자형(초/밀리초) 처리
        if pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            med = float(num.dropna().median()) if num.dropna().size else 0.0
            unit = "ms" if med >= 1e12 else "s"
            ts = pd.to_datetime(num, unit=unit, errors="coerce", utc=True)
            return ts.dt.tz_convert("Asia/Seoul")

        # 문자열 혼합 처리(자동 추정)
        ss = s.astype(str).str.strip()
        ts = pd.to_datetime(ss, errors="coerce", utc=True, infer_datetime_format=True)
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
    if not os.path.isdir(md):
        return []
    syms = []
    for fn in os.listdir(md):
        if not (fn.endswith(".pt") or fn.endswith(".meta.json")):
            continue
        sym = fn.split("_", 1)[0].upper()
        if sym.endswith("USDT") and len(sym) >= 6:
            syms.append(sym)
    return sorted(set(syms), key=syms.index)

def _select_60(symbols):
    if len(symbols) >= 60:
        return symbols[:60]
    need = 60 - len(symbols)
    filler = [s for s in _BASELINE_SYMBOLS if s not in symbols][:need]
    return symbols + filler

def _compute_groups(symbols, group_size=5):
    return [symbols[i:i+group_size] for i in range(0, len(symbols), group_size)]

# ✅ [고정화] 심볼/그룹
SYMBOLS = list(_BASELINE_SYMBOLS)
SYMBOL_GROUPS = _compute_groups(SYMBOLS, 5)

SYMBOL_MAP["bybit"]  = {s: s for s in SYMBOLS}
SYMBOL_MAP["binance"] = {s: s for s in SYMBOLS}

def get_ALL_SYMBOLS(): return list(SYMBOLS)
def get_SYMBOL_GROUPS(): return list(SYMBOL_GROUPS)

# ========================= 그룹 순서 제어(지속성) =========================
_STATE_DIR = "/persistent/state"
_STATE_PATH = os.path.join(_STATE_DIR, "group_order.json")

class GroupOrderManager:
    """현재 그룹 인덱스/학습 완료 심볼 집합 관리 + 파일 지속성"""
    def __init__(self, groups: List[List[str]]):
        self.groups = [list(g) for g in groups]
        self.idx = 0
        self.trained = {}
        self._load()

    def _load(self):
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            if os.path.isfile(_STATE_PATH):
                st = json.load(open(_STATE_PATH, "r", encoding="utf-8"))
                saved_syms = st.get("symbols", [])
                saved_groups = _compute_groups(saved_syms, 5) if saved_syms else st.get("groups", [])
                if saved_groups:
                    self.groups = saved_groups
                self.idx = int(st.get("idx", 0))
                self.trained = {int(k): set(v) for k, v in st.get("trained", {}).items()}
                print(f"[🧭 그룹상태 로드] idx={self.idx}, trained_keys={list(self.trained.keys())}")
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
            }
            json.dump(payload, open(_STATE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[⚠️ 그룹상태 저장 실패] {e}")

    def current_index(self) -> int:
        return max(0, min(self.idx, max(0, len(self.groups) - 1)))

    def current_group(self) -> List[str]:
        i = self.current_index()
        return self.groups[i] if self.groups else []

    def should_train(self, symbol: str) -> bool:
        i = self.current_index()
        gset = set(self.current_group())
        done = self.trained.get(i, set())
        ok = (symbol in gset) and (symbol not in done)
        if not ok:
            where = "다음 그룹" if symbol not in gset else "이미 학습됨"
            print(f"[⛔ 순서강제] {symbol} → 현재 그룹{i} 차례 아님 ({where})")
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
        ready = group and group.issubset(done)
        if ready:
            print(f"[🚦 예측대기] 그룹{i} 전체 학습 완료 → 예측 실행 준비")
        return ready

    def mark_group_predicted(self):
        i = self.current_index()
        print(f"[✅ 예측완료] 그룹{i} → 다음 그룹으로 이동")
        self.idx = (i + 1) % max(1, len(self.groups))
        self.trained.setdefault(self.idx, set())
        self._save()

    def reset(self, start_index: int = 0):
        self.idx = max(0, min(start_index, max(0, len(self.groups) - 1)))
        self.trained = {self.idx: set()}
        self._save()
        print(f"[♻️ 그룹순서 리셋] idx={self.idx}")

    def rebuild_groups(self, symbols: Optional[List[str]] = None, group_size: int = 5):
        self.groups = _compute_groups(symbols or SYMBOLS, group_size)
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

# 🚑 초기화 안전장치: 모델이 없거나 강제 플래그가 있으면 그룹 상태 자동 리셋
def _models_exist(model_dir="/persistent/models"):
    try:
        if not os.path.isdir(model_dir):
            return False
        for fn in os.listdir(model_dir):
            full = os.path.join(model_dir, fn)
            if os.path.isdir(full):
                for _, _, files in os.walk(full):
                    if any(f.endswith((".pt", ".ptz", ".meta.json")) for f in files):
                        return True
            else:
                if fn.endswith((".pt", ".ptz", ".meta.json")):
                    return True
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
                print(f"[캐시 HIT] {key}")
                return cls._cache[key]
            else:
                print(f"[캐시 EXPIRED] {key}")
                cls.delete(key)
        return None

    @classmethod
    def set(cls, key, value):
        cls._cache[key] = value
        cls._ttl[key] = time.time()
        print(f"[캐시 SET] {key}")

    @classmethod
    def delete(cls, key):
        if key in cls._cache:
            del cls._cache[key]
            cls._ttl.pop(key, None)
            print(f"[캐시 DELETE] {key}")

    @classmethod
    def clear(cls):
        cls._cache.clear()
        cls._ttl.clear()
        print("[캐시 CLEAR ALL]")

def _auto_reset_group_state_if_needed():
    force = os.getenv("FORCE_RESET_GROUPS", "0") == "1"
    no_models = not _models_exist()
    if force or no_models:
        try:
            GROUP_MGR.reset(0)
            try:
                CacheManager.clear()
            except Exception:
                pass
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

# 미래 수익률
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
            if high[j] > mx:
                mx = high[j]
            j += 1  # ✅ 교착 방지
        j0 = max(j - 1, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        out[i] = float((mx - base) / (base + 1e-12))
    return out.astype(np.float32)

def future_gains(df: pd.DataFrame, strategy: str) -> np.ndarray:
    return future_gains_by_hours(df, {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24))

# 숫자 downcast
def _downcast_numeric(df: pd.DataFrame, prefer_float32: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
            if prefer_float32 and df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
    return df

# OHLC 일관성/극단치 보정
def _fix_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["open","high","low","close"]] = df[["open","high","low","close"]].replace([np.inf, -np.inf], np.nan)
    df["volume"] = df["volume"].replace([np.inf, -np.inf], np.nan)
    for c in ["open","high","low","close"]:
        df.loc[df[c] <= 0, c] = np.nan
    df.loc[df["volume"] < 0, "volume"] = np.nan
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].ffill()
    df = df.dropna(subset=["open","high","low","close","volume"])
    mx = df[["open","close","low"]].max(axis=1)
    mn = df[["open","close","high"]].min(axis=1)
    df["high"] = np.maximum(df["high"].values, mx.values)
    df["low"]  = np.minimum(df["low"].values,  mn.values)
    return df

def _winsorize_prices(df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    cols = ["open","high","low","close"]
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        lo = s.quantile(lower_q); hi = s.quantile(upper_q)
        df[c] = s.clip(lower=lo, upper=hi).astype(np.float32)
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce")
        lo = max(0.0, v.quantile(lower_q)); hi = v.quantile(upper_q)
        df["volume"] = v.clip(lower=lo, upper=hi).astype(np.float32)
    return df

# ========================= 데이터셋 생성 =========================
def _bin_labels(values: np.ndarray, num_classes: int) -> np.ndarray:
    """연속값을 균등 구간으로 잘라 0..num_classes-1 라벨로 매핑"""
    if len(values) == 0 or num_classes < 2:
        return np.zeros(len(values), dtype=np.int64)
    lo = float(np.nanmin(values)); hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(len(values), dtype=np.int64)
    edges = np.linspace(lo, hi, num_classes + 1, dtype=np.float64)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, num_classes - 1)
    return idx.astype(np.int64)

def create_dataset(features, window=10, strategy="단기", input_size=None):
    """
    features: list[dict], return: (X, y). 부족 시 더미 샘플 반환.
    - ✅ 라벨은 항상 현재 NUM_CLASSES에 맞춰 생성(동적 3클래스 금지)
    """
    import pandas as _pd
    from config import MIN_FEATURES, get_NUM_CLASSES

    def _dummy(symbol_name):
        from config import MIN_FEATURES as _MINF
        safe_failed_result(symbol_name, strategy, reason="create_dataset 입력 feature 부족/실패")
        X = np.zeros((max(1, window), window, input_size if input_size else _MINF), dtype=np.float32)
        y = np.zeros((max(1, window),), dtype=np.int64)
        return X, y

    num_classes = max(2, int(get_NUM_CLASSES()))
    symbol_name = "UNKNOWN"
    if isinstance(features, list) and features and isinstance(features[0], dict) and "symbol" in features[0]:
        symbol_name = features[0]["symbol"]

    if not isinstance(features, list) or len(features) <= window:
        print(f"[⚠️ 부족] features length={len(features) if isinstance(features, list) else 'Invalid'}, window={window}")
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
            print("[❌ feature_cols 없음]")
            return _dummy(symbol_name)

        from config import MIN_FEATURES as _MINF
        if len(feature_cols) < _MINF:
            for i in range(len(feature_cols), _MINF):
                df[f"pad_{i}"] = np.float32(0.0)
                feature_cols.append(f"pad_{i}")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols].astype(np.float32))
        df_s = pd.DataFrame(scaled.astype(np.float32), columns=feature_cols)
        df_s["timestamp"] = df["timestamp"].values
        df_s["high"] = df["high"] if "high" in df.columns else df["close"]

        if input_size and len(feature_cols) < input_size:
            for i in range(len(feature_cols), input_size):
                df_s[f"pad_{i}"] = np.float32(0.0)

        features = df_s.to_dict(orient="records")

        # ── 시퀀스 & 이득률 계산 ──
        strategy_minutes = {"단기": 240, "중기": 1440, "장기": 2880}
        lookahead = strategy_minutes.get(strategy, 1440)
        samples, gains = [], []
        row_cols = [c for c in df_s.columns if c != "timestamp"]

        for i in range(window, len(features)):
            seq = features[i - window:i]
            base = features[i]
            entry_time = pd.to_datetime(base.get("timestamp"), errors="coerce", utc=True).tz_convert("Asia/Seoul")
            entry_price = float(base.get("close", 0.0))
            if pd.isnull(entry_time) or entry_price <= 0:
                continue
            try:
                future = [
                    f for f in features[i + 1:]
                    if (pd.to_datetime(f.get("timestamp", None), utc=True) - entry_time) <= pd.Timedelta(minutes=lookahead)
                ]
            except Exception:
                continue
            vprices = [f.get("high", f.get("close", entry_price)) for f in future if f.get("high", 0) > 0]
            if len(seq) != window or not vprices:
                continue
            max_future = max(vprices)
            gain = float((max_future - entry_price) / (entry_price + 1e-6))
            gains.append(gain)

            sample = [[float(r.get(c, 0.0)) for c in row_cols] for r in seq]
            if input_size:
                for j in range(len(sample)):
                    row = sample[j]
                    if len(row) < input_size:
                        row.extend([0.0] * (input_size - len(row)))
                    elif len(row) > input_size:
                        sample[j] = row[:input_size]
            samples.append(sample)

        # 라벨 생성(항상 NUM_CLASSES)
        if samples:
            gains_arr = np.asarray(gains, dtype=np.float64)
            y = _bin_labels(gains_arr, num_classes)
            X = np.array(samples, dtype=np.float32)
            if len(X) != len(y):
                m = min(len(X), len(y))
                X = X[:m]; y = y[:m]
            print(f"[✅ create_dataset 완료] 샘플 수: {len(y)}, X.shape={X.shape}, NUM_CLASSES={num_classes}")
            return X, y

        # fallback: 인접 변화율 기반이지만 역시 NUM_CLASSES로 binning
        closes = df_s["close"].to_numpy(dtype=np.float32)
        pct = np.diff(closes) / (closes[:-1] + 1e-6)
        fb_samples, fb_labels = [], []
        for i in range(window, len(df_s) - 1):
            seq_rows = df_s.iloc[i - window:i]
            sample = [[float(r.get(c, 0.0)) for c in row_cols] for _, r in seq_rows.iterrows()]
            if input_size:
                for j in range(len(sample)):
                    row = sample[j]
                    if len(row) < input_size:
                        row.extend([0.0] * (input_size - len(row)))
                    elif len(row) > input_size:
                        sample[j] = row[:input_size]
            fb_samples.append(sample)
            fb_labels.append(pct[i] if i < len(pct) else 0.0)

        if not fb_samples:
            return _dummy(symbol_name)

        y = _bin_labels(np.asarray(fb_labels, dtype=np.float64), num_classes)
        X = np.array(fb_samples, dtype=np.float32)
        if len(X) != len(y):
            m = min(len(X), len(y))
            X = X[:m]; y = y[:m]
        print(f"[✅ create_dataset 완료] (fallback pct) 샘플 수: {len(y)}, X.shape={X.shape}, NUM_CLASSES={num_classes}")
        return X, y

    except Exception as e:
        print(f"[❌ 최상위 예외] create_dataset 실패 → {e}")
        return _dummy(symbol_name)

# ========================= 정제/클립 =========================
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = _parse_ts_series(df["timestamp"])
    elif "time" in df.columns:
        df["timestamp"] = _parse_ts_series(df["time"])
    else:
        df["timestamp"] = _parse_ts_series(pd.Series([pd.NaT] * len(df)))

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = df["timestamp"]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df = _downcast_numeric(df)
    df = _fix_ohlc_consistency(df)
    df = _winsorize_prices(df, 0.001, 0.999)
    return df[cols]

def _clip_tail(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
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
    real_symbol = SYMBOL_MAP["bybit"].get(symbol, symbol)
    target_rows = int(limit)
    collected, total, last_oldest = [], 0, None

    interval = _map_bybit_interval(interval)

    while total < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total
                req = min(1000, rows_needed)
                params = {"category": "linear", "symbol": real_symbol, "interval": interval, "limit": req}
                if end_time is not None:
                    # Bybit end: ms(UTC)
                    params["end"] = int(pd.to_datetime(end_time).tz_convert("UTC").timestamp() * 1000)

                print(f"[📡 Bybit 요청] {real_symbol}-{interval} | 시도 {attempt+1}/{max_retry} | 요청 수량={req} | end={end_time}")
                res = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10, headers=REQUEST_HEADERS)
                res.raise_for_status()
                data = res.json()
                raw = (data or {}).get("result", {}).get("list", [])
                if not raw:
                    print(f"[❌ 데이터 없음] {real_symbol} (시도 {attempt+1})")
                    break

                if isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6:
                    df_chunk = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(raw[0])])
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
                # Bybit end 파라미터는 밀리초 UTC 기준 → UTC로 맞춰 뒤로 이동
                end_time = pd.to_datetime(oldest_ts).tz_convert("UTC") - pd.Timedelta(milliseconds=1)
                time.sleep(0.2)
                break
            except RequestException as e:
                print(f"[에러] get_kline({real_symbol}) 네트워크 실패 → {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"[에러] get_kline({real_symbol}) 실패 → {e}")
                time.sleep(0.5)
                continue
        if not success:
            break

    if collected:
        df = _normalize_df(pd.concat(collected, ignore_index=True))
        print(f"[📊 수집 완료] {symbol}-{interval} → 총 {len(df)}개 봉 확보")
        return df
    print(f"[❌ 최종 실패] {symbol}-{interval} → 수집된 봉 없음")
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 거래소 수집기(Binance) =========================
def get_kline_binance(symbol: str, interval: str = "240", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    real_symbol = SYMBOL_MAP["binance"].get(symbol, symbol)
    _bin_iv = None
    for _, cfg in STRATEGY_CONFIG.items():
        if cfg.get("interval") == interval:
            _bin_iv = cfg.get("binance_interval")
            break
    if _bin_iv is None:
        _bin_iv = {"240": "4h", "D": "1d", "2D": "2d", "60": "1h"}.get(interval, "1h")

    target_rows = int(limit)
    collected, total, last_oldest = [], 0, None

    if not BINANCE_ENABLED or _is_binance_blocked():
        print("[⛔ Binance 비활성화 상태] 환경변수 또는 일시차단")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

    while total < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total
                req = min(1000, rows_needed)
                params = {"symbol": real_symbol, "interval": _bin_iv, "limit": req}
                if end_time is not None:
                    # Binance endTime: ms(UTC)
                    params["endTime"] = int(pd.to_datetime(end_time).tz_convert("UTC").timestamp() * 1000)
                print(f"[📡 Binance 요청] {real_symbol}-{interval} | 요청 {req}개 | 시도 {attempt+1}/{max_retry} | end_time={end_time}")

                res = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/klines", params=params, timeout=10, headers=REQUEST_HEADERS)
                try:
                    res.raise_for_status()
                except HTTPError as he:
                    if getattr(he.response, "status_code", None) == 418:
                        print("[🚨 Binance 418 감지] 자동 백오프 및 폴백 비활성화")
                        _block_binance_for(1800)
                        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])
                    raise

                raw = res.json()
                if not raw:
                    print(f"[❌ Binance 데이터 없음] {real_symbol}-{interval} (시도 {attempt+1})")
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
                # Binance도 endTime은 UTC ms
                end_time = pd.to_datetime(oldest_ts).tz_convert("UTC") - pd.Timedelta(milliseconds=1)
                time.sleep(0.3)
                break
            except RequestException as e:
                print(f"[에러] get_kline_binance({real_symbol}) 네트워크 실패 → {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"[에러] get_kline_binance({real_symbol}) 실패 → {e}")
                time.sleep(0.5)
                continue
        if not success:
            break

    if collected:
        df = _normalize_df(pd.concat(collected, ignore_index=True))
        print(f"[📊 Binance 수집 완료] {symbol}-{interval} → 총 {len(df)}개 봉 확보")
        return df
    print(f"[❌ 최종 실패] {symbol}-{interval} → 수집된 봉 없음")
    return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 통합 수집 + 병합 =========================
def get_merged_kline_by_strategy(symbol: str, strategy: str) -> pd.DataFrame:
    config = STRATEGY_CONFIG.get(strategy)
    if not config:
        print(f"[❌ 실패] 전략 설정 없음: {strategy}")
        return pd.DataFrame()
    interval = config["interval"]; base_limit = int(config["limit"]); max_total = base_limit

    def fetch_until_target(fetch_func, src):
        total = []; end = None; cnt = 0; max_rep = 10
        print(f"[⏳ {src} 데이터 수집 시작] {symbol}-{strategy} | 목표 {base_limit}개")
        while cnt < max_total and len(total) < max_rep:
            dfc = fetch_func(symbol, interval=interval, limit=base_limit, end_time=end)
            if dfc is None or dfc.empty:
                break
            total.append(dfc); cnt += len(dfc)
            if len(dfc) < base_limit:
                break
            oldest = dfc["timestamp"].min()
            # 통일: 다음 요청 end는 UTC ms 기준
            end = pd.to_datetime(oldest).tz_convert("UTC") - pd.Timedelta(milliseconds=1)
        dff = _normalize_df(pd.concat(total, ignore_index=True)) if total else pd.DataFrame()
        print(f"[✅ {src} 수집 완료] {symbol}-{strategy} → {len(dff)}개")
        return dff

    df_bybit = fetch_until_target(get_kline, "Bybit")
    df_binance = pd.DataFrame()
    if len(df_bybit) < base_limit and BINANCE_ENABLED and not _is_binance_blocked():
        print(f"[⏳ Binance 보충 시작] 부족 {base_limit - len(df_bybit)}개")
        df_binance = fetch_until_target(get_kline_binance, "Binance")

    df_all = _normalize_df(pd.concat([df_bybit, df_binance], ignore_index=True)) if (not df_bybit.empty or not df_binance.empty) else pd.DataFrame()
    if df_all.empty:
        print(f"[⏩ 학습 스킵] {symbol}-{strategy} → 거래소 데이터 전무")
        return pd.DataFrame()

    df_all = _clip_tail(df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), base_limit)
    for c in ["timestamp","open","high","low","close","volume"]:
        if c not in df_all.columns:
            df_all[c] = 0.0 if c != "timestamp" else pd.Timestamp.now(tz="Asia/Seoul")
    df_all.attrs["augment_needed"] = len(df_all) < base_limit
    print(f"[🔄 병합 완료] {symbol}-{strategy} → 최종 {len(df_all)}개 (목표 {base_limit}개)")
    if len(df_all) < base_limit:
        print(f"[⚠️ 경고] {symbol}-{strategy} 데이터 부족 ({len(df_all)}/{base_limit})")
    return df_all

# ========================= 전략별 Kline (캐시 포함) =========================
def get_kline_by_strategy(symbol: str, strategy: str):
    cache_key = f"{symbol}-{strategy}"
    cached = CacheManager.get(cache_key, ttl_sec=600)
    if cached is not None:
        print(f"[✅ 캐시 사용] {symbol}-{strategy} → {len(cached)}개 봉")
        return cached
    try:
        cfg = STRATEGY_CONFIG.get(strategy, {"limit": 300, "interval": "D"})
        limit = int(cfg.get("limit", 300)); interval = cfg.get("interval", "D")

        df_bybit = []; total_bybit = 0; end = None
        print(f"[📡 Bybit 1차 반복 수집 시작] {symbol}-{strategy} (limit={limit})")
        while total_bybit < limit:
            dfc = get_kline(symbol, interval=interval, limit=limit, end_time=end)
            if dfc is None or dfc.empty:
                break
            df_bybit.append(dfc); total_bybit += len(dfc)
            oldest = dfc["timestamp"].min()
            end = pd.to_datetime(oldest).tz_convert("UTC") - pd.Timedelta(milliseconds=1)
            if len(dfc) < limit:
                break
        df_bybit = _normalize_df(pd.concat(df_bybit, ignore_index=True)) if df_bybit else pd.DataFrame()

        df_binance = []; total_binance = 0
        if len(df_bybit) < int(limit * 0.9) and BINANCE_ENABLED and not _is_binance_blocked():
            print(f"[📡 Binance 2차 반복 수집 시작] {symbol}-{strategy} (limit={limit})")
            end = None
            while total_binance < limit:
                try:
                    dfc = get_kline_binance(symbol, interval=interval, limit=limit, end_time=end)
                    if dfc is None or dfc.empty:
                        break
                    df_binance.append(dfc); total_binance += len(dfc)
                    oldest = dfc["timestamp"].min()
                    end = pd.to_datetime(oldest).tz_convert("UTC") - pd.Timedelta(milliseconds=1)
                    if len(dfc) < limit:
                        break
                except Exception as be:
                    print(f"[❌ Binance 수집 실패] {symbol}-{strategy} → {be}")
                    break
        elif len(df_bybit) < int(limit * 0.9):
            print("[⛔ Binance 폴백 스킵] 비활성화 또는 일시차단 상태")

        df_binance = _normalize_df(pd.concat(df_binance, ignore_index=True)) if df_binance else pd.DataFrame()

        df_list = [d for d in [df_bybit, df_binance] if d is not None and not d.empty]
        df = _normalize_df(pd.concat(df_list, ignore_index=True)) if df_list else pd.DataFrame()
        df = _clip_tail(df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), limit)

        total = len(df); min_required = max(60, int(limit * 0.90))
        if total < min_required:
            print(f"[⚠️ 수집 수량 부족] {symbol}-{strategy} → 총 {total}개 (최소보장 {min_required}, 목표 {limit}) → 통합 재시도")
            df_retry = get_merged_kline_by_strategy(symbol, strategy)
            if not df_retry.empty and len(df_retry) > total:
                df = _clip_tail(df_retry, limit); total = len(df)

        if total < min_required:
            print(f"[🚨 최종 부족] {symbol}-{strategy} → {total}/{min_required} (학습/예측 영향 가능)")
        else:
            print(f"[✅ 수집 성공] {symbol}-{strategy} → 총 {total}개")

        df.attrs["augment_needed"] = total < limit
        df.attrs["enough_for_training"] = total >= min_required
        CacheManager.set(cache_key, df)
        return df
    except Exception as e:
        print(f"[❌ 데이터 수집 실패] {symbol}-{strategy} → {e}")
        safe_failed_result(symbol, strategy, reason=str(e))
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])

# ========================= 프리패치 =========================
def prefetch_symbol_groups(strategy: str):
    for group in SYMBOL_GROUPS:
        for sym in group:
            try:
                get_kline_by_strategy(sym, strategy)
            except Exception as e:
                print(f"[⚠️ prefetch 실패] {sym}-{strategy}: {e}")

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

# ========================= 피처 생성 (+레짐) =========================
_feature_cache = {}
def compute_features(symbol: str, df: pd.DataFrame, strategy: str, required_features: list = None, fallback_input_size: int = None) -> pd.DataFrame:
    cache_key = f"{symbol}-{strategy}-features"
    cached = CacheManager.get(cache_key, ttl_sec=600)
    if cached is not None:
        print(f"[캐시 HIT] {cache_key}")
        return cached

    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        print(f"[❌ compute_features 실패] 입력 DataFrame empty or invalid")
        safe_failed_result(symbol, strategy, reason="입력DataFrame empty")
        return pd.DataFrame()

    df = df.copy()
    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now", utc=True).tz_convert("Asia/Seoul")
    df["strategy"] = strategy

    base_cols = ["open","high","low","close","volume"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[["timestamp"] + base_cols]

    if len(df) < 20:
        print(f"[⚠️ 피처 실패] {symbol}-{strategy} → row 수 부족: {len(df)}")
        safe_failed_result(symbol, strategy, reason=f"row 부족 {len(df)}")
        return df

    try:
        # 기본 지표
        df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["bollinger"] = df["close"].rolling(window=20, min_periods=1).std()
        df["volatility"] = df["high"] - df["low"]
        df["trend_score"] = (df["close"] > df["ma20"]).astype(int)
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["roc"] = df["close"].pct_change(periods=10)

        # ta 라이브러리 확장(가능 시)
        try:
            import ta as _ta
            df["adx"] = _ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
            df["cci"] = _ta.trend.cci(df["high"], df["low"], df["close"], window=20, fillna=True)
            df["mfi"] = _ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14, fillna=True)
            df["obv"] = _ta.volume.on_balance_volume(df["close"], df["volume"], fillna=True)
            df["atr"] = _ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14, fillna=True)
            df["williams_r"] = _ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14, fillna=True)
            df["stoch_k"] = _ta.momentum.stoch(df["high"], df["low"], df["close"], fillna=True)
            df["stoch_d"] = _ta.momentum.stoch_signal(df["high"], df["low"], df["close"], fillna=True)
        except Exception:
            pass

        df["vwap"] = (df["volume"] * df["close"]).cumsum() / (df["volume"].cumsum() + 1e-6)

        # 레짐(옵션)
        regime_cfg = get_REGIME()
        if regime_cfg.get("enabled", False):
            try:
                import ta as _ta2
                atr_win = int(regime_cfg.get("atr_window", 14))
                trend_win = int(regime_cfg.get("trend_window", 50))
                vol_high = float(regime_cfg.get("vol_high_pct", 0.9))
                vol_low  = float(regime_cfg.get("vol_low_pct", 0.5))
                df["atr_val"] = _ta2.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_win, fillna=True)
                thr_high = df["atr_val"].quantile(vol_high)
                thr_low  = df["atr_val"].quantile(vol_low)
                df["vol_regime"] = np.where(df["atr_val"] >= thr_high, 2, np.where(df["atr_val"] <= thr_low, 0, 1))
                df["ma_trend"] = df["close"].rolling(window=trend_win, min_periods=1).mean()
                slope = df["ma_trend"].diff()
                df["trend_regime"] = np.where(slope > 0, 2, np.where(slope < 0, 0, 1))
                df["regime_tag"] = df["vol_regime"] * 3 + df["trend_regime"]
            except Exception:
                pass

        # 정리/스케일
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        feat_cols = [c for c in df.columns if c != "timestamp"]

        from config import FEATURE_INPUT_SIZE as _FIS
        if len(feat_cols) < _FIS:
            for i in range(len(feat_cols), _FIS):
                df[f"pad_{i}"] = 0.0
                feat_cols.append(f"pad_{i}")

        df[feat_cols] = _downcast_numeric(df[feat_cols]).astype(np.float32)
        df[feat_cols] = MinMaxScaler().fit_transform(df[feat_cols])
    except Exception as e:
        print(f"[❌ compute_features 실패] feature 계산 예외 → {e}")
        safe_failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        return df

    if df.empty or df.isnull().values.any():
        print(f"[❌ compute_features 실패] 결과 DataFrame 문제 → 빈 df 또는 NaN 존재")
        safe_failed_result(symbol, strategy, reason="최종 결과 DataFrame 오류")
        return df

    print(f"[✅ 완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    print(f"[🔍 feature 상태] {symbol}-{strategy} → shape: {df.shape}, NaN: {df.isnull().values.any()}, 컬럼수: {len(df.columns)}")
    CacheManager.set(cache_key, df)
    return df
