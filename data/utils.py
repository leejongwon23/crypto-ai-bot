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

# =========================
# 기본 상수/전역
# =========================
BASE_URL = "https://api.bybit.com"
BINANCE_BASE_URL = "https://fapi.binance.com"  # Binance Futures (USDT-M)
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}

# --- 기본(백업) 심볼 시드 60개: 최후 fallback 용 ---
_BASELINE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "TRXUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "ATOMUSDT", "XLMUSDT",
    "ETCUSDT", "ICPUSDT", "HBARUSDT", "FILUSDT", "SANDUSDT",
    "RNDRUSDT", "INJUSDT", "NEARUSDT", "APEUSDT", "ARBUSDT",
    "AAVEUSDT", "OPUSDT", "SUIUSDT", "DYDXUSDT", "CHZUSDT",
    "LDOUSDT", "STXUSDT", "GMTUSDT", "FTMUSDT", "WLDUSDT",
    "TIAUSDT", "SEIUSDT", "ARKMUSDT", "JASMYUSDT", "AKTUSDT",
    "GMXUSDT", "SKLUSDT", "BLURUSDT", "ENSUSDT", "CFXUSDT",
    "FLOWUSDT", "ALGOUSDT", "MINAUSDT", "NEOUSDT", "MASKUSDT",
    "KAVAUSDT", "BATUSDT", "ZILUSDT", "WAVESUSDT", "OCEANUSDT",
    "1INCHUSDT", "YFIUSDT", "STGUSDT", "GALAUSDT", "IMXUSDT"
]

# ✅ 거래소별 심볼 매핑 (초기값은 빈 dict; 아래 discover 이후 채움)
SYMBOL_MAP = {"bybit": {}, "binance": {}}

# ✅ 전략 설정은 단일 소스(config.py)를 따른다 + 레짐 옵션(get_REGIME) 안전 임포트
try:
    from config import STRATEGY_CONFIG, get_REGIME  # {"단기":{"interval":"240","limit":1000,"binance_interval":"4h"}, ...}
except Exception:
    STRATEGY_CONFIG = {
        "단기": {"interval": "240", "limit": 1000, "binance_interval": "4h"},
        "중기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
        "장기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
    }
    def get_REGIME():
        return {
            "enabled": False,
            "atr_window": 14,
            "rsi_window": 14,
            "trend_window": 50,
            "vol_high_pct": 0.9,
            "vol_low_pct": 0.5
        }

# =========================
# 심볼 동적 발견 + 60개 고정화
# =========================
def _merge_unique(*lists):
    seen = set()
    out = []
    for L in lists:
        for x in L:
            if not x:
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out

def _discover_from_env():
    # 우선순위 1) ENV: PREDICT_SYMBOLS 또는 SYMBOLS_OVERRIDE (쉼표/공백 구분 허용)
    raw = os.getenv("PREDICT_SYMBOLS") or os.getenv("SYMBOLS_OVERRIDE") or ""
    if not raw.strip():
        return []
    parts = [p.strip().upper() for p in raw.replace("\n", ",").replace(" ", ",").split(",") if p.strip()]
    return parts

def _discover_from_models():
    # 우선순위 2) /persistent/models 스캔: {SYMBOL}_{전략}_*.pt / .meta.json
    model_dir = "/persistent/models"
    if not os.path.isdir(model_dir):
        return []
    syms = []
    for fn in os.listdir(model_dir):
        if not (fn.endswith(".pt") or fn.endswith(".meta.json")):
            continue
        # 파일명 규약: SYMBOL_STRATEGY_*.* → 언더스코어 첫 토큰
        sym = fn.split("_", 1)[0].upper()
        # 간단 검증: 선물 USDT 마켓 패턴
        if sym.endswith("USDT") and len(sym) >= 6:
            syms.append(sym)
    return sorted(set(syms), key=syms.index)

def _select_60(symbols):
    # 정확히 60개로 맞춤(부족하면 baseline로 채움, 초과면 앞에서 60개)
    if len(symbols) >= 60:
        return symbols[:60]
    need = 60 - len(symbols)
    filler = [s for s in _BASELINE_SYMBOLS if s not in symbols][:need]
    return symbols + filler

def _compute_groups(symbols, group_size=5):
    return [symbols[i:i+group_size] for i in range(0, len(symbols), group_size)]

# --- 실제 심볼 집합 계산 ---
_env_syms   = _discover_from_env()
_model_syms = _discover_from_models()
SYMBOLS = _select_60(_merge_unique(_env_syms, _model_syms, _BASELINE_SYMBOLS))
SYMBOL_GROUPS = _compute_groups(SYMBOLS, group_size=5)

# 거래소 맵 갱신
SYMBOL_MAP["bybit"]   = {s: s for s in SYMBOLS}
SYMBOL_MAP["binance"] = {s: s for s in SYMBOLS}

# 외부 공개 함수 (관우/트리거/백엔드 공통 사용)
def get_ALL_SYMBOLS():
    return list(SYMBOLS)

def get_SYMBOL_GROUPS():
    return list(SYMBOL_GROUPS)

# =========================
# 캐시 매니저 (이 파일 내부 사용)
# =========================
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

# =========================
# 실패 로깅(순환 의존 제거용 경량 헬퍼)
# =========================
def safe_failed_result(symbol, strategy, reason=""):
    try:
        from failure_db import insert_failure_record  # 순환 없음
        payload = {
            "symbol": symbol or "UNKNOWN",
            "strategy": strategy or "UNKNOWN",
            "model": "utils",
            "reason": reason,
            "timestamp": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_class": -1,
            "label": -1
        }
        insert_failure_record(payload, feature_vector=[])
    except Exception as e:
        print(f"[⚠️ safe_failed_result 실패] {e}")

# =========================
# 기타 유틸
# =========================
def get_btc_dominance():
    global BTC_DOMINANCE_CACHE
    now = time.time()
    if now - BTC_DOMINANCE_CACHE["timestamp"] < 1800:
        return BTC_DOMINANCE_CACHE["value"]
    try:
        url = "https://api.coinpaprika.com/v1/global"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        dom = float(data["bitcoin_dominance_percentage"]) / 100
        BTC_DOMINANCE_CACHE = {"value": round(dom, 4), "timestamp": now}
        return BTC_DOMINANCE_CACHE["value"]
    except:
        return BTC_DOMINANCE_CACHE["value"]

# =========================
# ✅ 공용: 미래 수익률 계산기
# (타임스탬프 파싱 고정: utc=True → Asia/Seoul)
# =========================
def future_gains_by_hours(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df is None or len(df) == 0 or "timestamp" not in df.columns:
        return np.zeros(0 if df is None else len(df), dtype=np.float32)

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
    close = pd.to_numeric(df["close"], errors="coerce").astype(np.float32).values
    high = pd.to_numeric((df["high"] if "high" in df.columns else df["close"]), errors="coerce").astype(np.float32).values

    out = np.zeros(len(df), dtype=np.float32)
    horizon = pd.Timedelta(hours=int(horizon_hours))

    j_start = 0
    for i in range(len(df)):
        t0 = ts.iloc[i]
        t1 = t0 + horizon
        j = max(j_start, i)
        max_h = high[i]
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > max_h:
                max_h = high[j]
            j += 1
        j_start = max(j_start, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        out[i] = float((max_h - base) / (base + 1e-12))
    return out.astype(np.float32)

def future_gains(df: pd.DataFrame, strategy: str) -> np.ndarray:
    horizon = {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)
    return future_gains_by_hours(df, horizon)

# =========================
# 공통: 숫자 downcast 유틸 (메모리 절약)
# =========================
def _downcast_numeric(df: pd.DataFrame, prefer_float32: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
            if prefer_float32 and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
    return df

# =========================
# 데이터셋 생성 (반환값 항상 X, y)
# =========================
def create_dataset(features, window=10, strategy="단기", input_size=None):
    """
    features: list[dict] (timestamp, open/high/low/close/volume, …)
    window:   시퀀스 길이
    return:   (X, y) — 항상 2개 반환
    """
    import pandas as pd
    from config import MIN_FEATURES
    # 🔧 변경: 예측 로그 오염 방지를 위해 log_prediction 호출 제거

    def _dummy(symbol_name):
        # 🔧 변경: 실패는 failure_db로만 기록하여 예측 로그와 분리
        safe_failed_result(symbol_name, strategy, reason="create_dataset 입력 feature 부족/실패")
        X = np.zeros((max(1, window), window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        y = np.zeros((max(1, window),), dtype=np.int64)
        return X, y

    symbol_name = "UNKNOWN"
    if isinstance(features, list) and features and isinstance(features[0], dict) and "symbol" in features[0]:
        symbol_name = features[0]["symbol"]

    if not isinstance(features, list) or len(features) <= window:
        print(f"[⚠️ 부족] features length={len(features) if isinstance(features, list) else 'Invalid'}, window={window}")
        return _dummy(symbol_name)

    try:
        df = pd.DataFrame(features)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.drop(columns=["strategy"], errors="ignore")

        # 숫자 칼럼 downcast
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
                pad_col = f"pad_{i}"
                df[pad_col] = np.float32(0.0)
                feature_cols.append(pad_col)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols].astype(np.float32))
        df_scaled = pd.DataFrame(scaled.astype(np.float32), columns=feature_cols)
        df_scaled["timestamp"] = df["timestamp"].values
        df_scaled["high"] = df["high"] if "high" in df.columns else df["close"]

        if input_size and len(feature_cols) < input_size:
            for i in range(len(feature_cols), input_size):
                pad_col = f"pad_{i}"
                df_scaled[pad_col] = np.float32(0.0)

        features = df_scaled.to_dict(orient="records")

        # lookahead 기반 라벨링
        strategy_minutes = {"단기": 240, "중기": 1440, "장기": 2880}
        lookahead_minutes = strategy_minutes.get(strategy, 1440)

        samples, gains = [], []
        for i in range(window, len(features)):
            seq = features[i - window:i]
            base = features[i]
            entry_time = pd.to_datetime(base.get("timestamp"), errors="coerce", utc=True).tz_convert("Asia/Seoul")
            entry_price = float(base.get("close", 0.0))

            if pd.isnull(entry_time) or entry_price <= 0:
                continue

            future = [f for f in features[i + 1:]
                      if pd.to_datetime(f.get("timestamp", None), utc=True) - entry_time <= pd.Timedelta(minutes=lookahead_minutes)]
            valid_prices = [f.get("high", f.get("close", entry_price)) for f in future if f.get("high", 0) > 0]
            if len(seq) != window or not valid_prices:
                continue

            max_future_price = max(valid_prices)
            gain = float((max_future_price - entry_price) / (entry_price + 1e-6))
            gains.append(gain)

            row_cols = [c for c in df_scaled.columns if c != "timestamp"]
            sample = [[float(r.get(c, 0.0)) for c in row_cols] for r in seq]

            if input_size:
                for j in range(len(sample)):
                    row = sample[j]
                    if len(row) < input_size:
                        row.extend([0.0] * (input_size - len(row)))
                    elif len(row) > input_size:
                        sample[j] = row[:input_size]
            samples.append((sample, gain))

        if not samples:
            print("[ℹ️ lookahead 기반 샘플 없음 → 인접 변화율로 3‑클래스 라벨링 사용]")
            closes_np = df_scaled["close"].to_numpy(dtype=np.float32)
            pct = np.diff(closes_np) / (closes_np[:-1] + 1e-6)
            thresh = 0.001  # ±0.1%

            for i in range(window, len(df_scaled) - 1):
                seq_rows = df_scaled.iloc[i - window:i]
                g = pct[i] if i < len(pct) else 0.0
                cls = 2 if g > thresh else (0 if g < -thresh else 1)

                row_cols = [c for c in df_scaled.columns if c != "timestamp"]
                sample = [[float(r.get(c, 0.0)) for c in row_cols] for _, r in seq_rows.iterrows()]
                if input_size:
                    for j in range(len(sample)):
                        row = sample[j]
                        if len(row) < input_size:
                            row.extend([0.0] * (input_size - len(row)))
                        elif len(row) > input_size:
                            sample[j] = row[:input_size]
                samples.append((sample, cls))

            X = np.array([s[0] for s in samples], dtype=np.float32)
            y = np.array([s[1] for s in samples], dtype=np.int64)
            if len(X) == 0:
                return _dummy(symbol_name)
            print(f"[✅ create_dataset 완료] (fallback 3‑class) 샘플 수: {len(y)}, X.shape={X.shape}")
            return X, y

        min_gain, max_gain = min(gains), max(gains)
        spread = max_gain - min_gain
        est_class = int(spread / 0.01)
        num_classes = max(3, min(21, est_class if est_class > 0 else 3))
        step = spread / num_classes if num_classes > 0 else 1e-6
        if step == 0:
            step = 1e-6

        X_list, y_list = [], []
        for sample, gain in samples:
            cls = min(int((gain - min_gain) / step), num_classes - 1)
            X_list.append(sample)
            y_list.append(cls)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        print(f"[✅ create_dataset 완료] 샘플 수: {len(y)}, X.shape={X.shape}, 동적 클래스 수: {num_classes}")
        return X, y

    except Exception as e:
        print(f"[❌ 최상위 예외] create_dataset 실패 → {e}")
        return _dummy(symbol_name)

# =========================
# 내부 헬퍼 (정제/클립)
# =========================
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """타임존/형/정렬/중복 제거 표준화 + 숫자 downcast"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","datetime"])
    df = df.copy()

    # timestamp 표준화 (항상 utc=True 후 Asia/Seoul로 변환)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"], errors="coerce", utc=True)
    else:
        ts = pd.NaT
    df["timestamp"] = pd.to_datetime(ts).dt.tz_convert("Asia/Seoul")

    # 필수 수치형
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df = df.dropna(subset=["timestamp","open","high","low","close","volume"])
    df["datetime"] = df["timestamp"]

    # 정렬/중복 제거
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 숫자 downcast (float32/정수 downcast)
    df = _downcast_numeric(df)

    return df[["timestamp","open","high","low","close","volume","datetime"]]

def _clip_tail(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """최신 limit개만 유지, 타임스탬프 역행 방지"""
    if df is None or df.empty:
        return df
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    # 역행 방지 (혹시 있을 역행 행 제거)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
    mask = ts.diff().fillna(pd.Timedelta(seconds=0)) >= pd.Timedelta(seconds=0)
    if not mask.all():
        df = df[mask].reset_index(drop=True)
    return df

# =========================
# 거래소 수집기(Bybit)
# =========================
def get_kline(symbol: str, interval: str = "60", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    real_symbol = SYMBOL_MAP["bybit"].get(symbol, symbol)
    target_rows = int(limit)
    collected_data, total_rows = [], 0
    last_oldest = None  # 🔧 추가: 무진행(같은 최저 ts 반복) 방지

    while total_rows < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total_rows
                request_limit = min(1000, rows_needed)
                params = {
                    "category": "linear",
                    "symbol": real_symbol,
                    "interval": interval,
                    "limit": request_limit
                }
                if end_time is not None:
                    params["end"] = int(end_time.timestamp() * 1000)

                print(f"[📡 Bybit 요청] {real_symbol}-{interval} | 시도 {attempt+1}/{max_retry} | 요청 수량={request_limit} | end={end_time}")
                res = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
                res.raise_for_status()
                data = res.json()

                if "result" not in data or "list" not in data["result"] or not data["result"]["list"]:
                    print(f"[❌ 데이터 없음] {real_symbol} (시도 {attempt+1})")
                    break

                raw = data["result"]["list"]
                if not raw or len(raw[0]) < 6:
                    print(f"[❌ 필드 부족] {real_symbol}")
                    break

                df_chunk = pd.DataFrame(raw, columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "turnover"
                ])
                df_chunk = _normalize_df(df_chunk)

                if df_chunk.empty:
                    break

                collected_data.append(df_chunk)
                total_rows += len(df_chunk)
                success = True

                if total_rows >= target_rows:
                    break

                oldest_ts = df_chunk["timestamp"].min()
                if last_oldest is not None and pd.to_datetime(oldest_ts) >= pd.to_datetime(last_oldest):
                    # 🔧 같은 경계 ts가 반복되면 더 과거로 강제 점프
                    oldest_ts = pd.to_datetime(oldest_ts) - pd.Timedelta(minutes=1)
                last_oldest = oldest_ts
                end_time = pd.to_datetime(oldest_ts).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)

                time.sleep(0.2)
                break

            except Exception as e:
                print(f"[에러] get_kline({real_symbol}) 실패 → {e}")
                time.sleep(1)
                continue

        if not success:
            break

    if collected_data:
        df = _normalize_df(pd.concat(collected_data, ignore_index=True))
        print(f"[📊 수집 완료] {symbol}-{interval} → 총 {len(df)}개 봉 확보")
        return df
    else:
        print(f"[❌ 최종 실패] {symbol}-{interval} → 수집된 봉 없음")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])

# =========================
# 거래소 수집기(Binance)
# =========================
def get_kline_binance(symbol: str, interval: str = "240", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    real_symbol = SYMBOL_MAP["binance"].get(symbol, symbol)

    # config의 binance_interval 우선 사용
    _bin_iv = None
    for name, cfg in STRATEGY_CONFIG.items():
        if cfg.get("interval") == interval:
            _bin_iv = cfg.get("binance_interval")
            break
    if _bin_iv is None:
        interval_map = {"240": "4h", "D": "1d", "2D": "2d", "60": "1h"}
        _bin_iv = interval_map.get(interval, "1h")

    target_rows = int(limit)
    collected_data, total_rows = [], 0
    last_oldest = None  # 🔧 추가: 무진행 방지

    while total_rows < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total_rows
                request_limit = min(1000, rows_needed)
                params = {
                    "symbol": real_symbol,
                    "interval": _bin_iv,
                    "limit": request_limit
                }
                if end_time is not None:
                    params["endTime"] = int(end_time.timestamp() * 1000)

                print(f"[📡 Binance 요청] {real_symbol}-{interval} | 요청 {request_limit}개 | 시도 {attempt+1}/{max_retry} | end_time={end_time}")
                res = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/klines", params=params, timeout=10)
                res.raise_for_status()
                raw = res.json()
                if not raw:
                    print(f"[❌ Binance 데이터 없음] {real_symbol}-{interval} (시도 {attempt+1})")
                    break

                df_chunk = pd.DataFrame(raw, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
                ])
                df_chunk = _normalize_df(df_chunk)

                if df_chunk.empty:
                    break

                collected_data.append(df_chunk)
                total_rows += len(df_chunk)
                success = True

                if total_rows >= target_rows:
                    break

                oldest_ts = df_chunk["timestamp"].min()
                if last_oldest is not None and pd.to_datetime(oldest_ts) >= pd.to_datetime(last_oldest):
                    oldest_ts = pd.to_datetime(oldest_ts) - pd.Timedelta(minutes=1)
                last_oldest = oldest_ts
                end_time = pd.to_datetime(oldest_ts).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)

                time.sleep(0.2)
                break

            except Exception as e:
                print(f"[에러] get_kline_binance({real_symbol}) 실패 → {e}")
                time.sleep(1)
                continue

        if not success:
            break

    if collected_data:
        df = _normalize_df(pd.concat(collected_data, ignore_index=True))
        print(f"[📊 Binance 수집 완료] {symbol}-{interval} → 총 {len(df)}개 봉 확보")
        return df
    else:
        print(f"[❌ 최종 실패] {symbol}-{interval} → 수집된 봉 없음")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])

# =========================
# (옵션) 통합 수집 + 병합 도우미
# =========================
def get_merged_kline_by_strategy(symbol: str, strategy: str) -> pd.DataFrame:
    config = STRATEGY_CONFIG.get(strategy)
    if not config:
        print(f"[❌ 실패] 전략 설정 없음: {strategy}")
        return pd.DataFrame()

    interval = config["interval"]
    base_limit = int(config["limit"])
    max_total = base_limit

    def fetch_until_target(fetch_func, source_name):
        total_data = []
        end_time = None
        total_count = 0
        max_repeat = 10

        print(f"[⏳ {source_name} 데이터 수집 시작] {symbol}-{strategy} | 목표 {base_limit}개")
        while total_count < max_total and len(total_data) < max_repeat:
            df_chunk = fetch_func(symbol, interval=interval, limit=base_limit, end_time=end_time)
            if df_chunk is None or df_chunk.empty:
                break

            total_data.append(df_chunk)
            total_count += len(df_chunk)

            if len(df_chunk) < base_limit:
                break

            oldest_ts = df_chunk["timestamp"].min()
            end_time = pd.to_datetime(oldest_ts).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)

        df_final = _normalize_df(pd.concat(total_data, ignore_index=True)) if total_data else pd.DataFrame()
        print(f"[✅ {source_name} 수집 완료] {symbol}-{strategy} → {len(df_final)}개")
        return df_final

    df_bybit = fetch_until_target(get_kline, "Bybit")
    df_binance = pd.DataFrame()
    if len(df_bybit) < base_limit:
        print(f"[⏳ Binance 보충 시작] 부족 {base_limit - len(df_bybit)}개")
        df_binance = fetch_until_target(get_kline_binance, "Binance")

    df_all = _normalize_df(pd.concat([df_bybit, df_binance], ignore_index=True))
    if df_all.empty:
        print(f"[⏩ 학습 스킵] {symbol}-{strategy} → 거래소 데이터 전무")
        return pd.DataFrame()

    df_all = _clip_tail(df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), base_limit)

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df_all.columns:
            df_all[col] = 0.0 if col != "timestamp" else pd.Timestamp.now(tz="Asia/Seoul")

    df_all.attrs["augment_needed"] = len(df_all) < base_limit
    print(f"[🔄 병합 완료] {symbol}-{strategy} → 최종 {len(df_all)}개 (목표 {base_limit}개)")
    if len(df_all) < base_limit:
        print(f"[⚠️ 경고] {symbol}-{strategy} 데이터 부족 ({len(df_all)}/{base_limit})")

    return df_all

# =========================
# 전략별 Kline 수집(캐시 포함, 기본 엔트리)
# =========================
def get_kline_by_strategy(symbol: str, strategy: str):
    cache_key = f"{symbol}-{strategy}"
    cached_df = CacheManager.get(cache_key, ttl_sec=600)
    if cached_df is not None:
        print(f"[✅ 캐시 사용] {symbol}-{strategy} → {len(cached_df)}개 봉")
        return cached_df

    try:
        cfg = STRATEGY_CONFIG.get(strategy, {"limit": 300, "interval": "D"})
        limit = int(cfg.get("limit", 300))
        interval = cfg.get("interval", "D")

        # 1) Bybit 반복 수집
        df_bybit = []
        total_bybit = 0
        end_time = None
        print(f"[📡 Bybit 1차 반복 수집 시작] {symbol}-{strategy} (limit={limit})")
        while total_bybit < limit:
            df_chunk = get_kline(symbol, interval=interval, limit=limit, end_time=end_time)
            if df_chunk is None or df_chunk.empty:
                break
            df_bybit.append(df_chunk)
            total_bybit += len(df_chunk)
            oldest_ts = df_chunk["timestamp"].min()
            end_time = pd.to_datetime(oldest_ts).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)
            if len(df_chunk) < limit:
                break

        df_bybit = _normalize_df(pd.concat(df_bybit, ignore_index=True)) if df_bybit else pd.DataFrame()

        # 2) Binance 보완 수집 (강화)
        df_binance = []
        total_binance = 0
        if len(df_bybit) < int(limit * 0.9):
            print(f"[📡 Binance 2차 반복 수집 시작] {symbol}-{strategy} (limit={limit})")
            end_time = None
            while total_binance < limit:
                try:
                    df_chunk = get_kline_binance(symbol, interval=interval, limit=limit, end_time=end_time)
                    if df_chunk is None or df_chunk.empty:
                        break
                    df_binance.append(df_chunk)
                    total_binance += len(df_chunk)
                    oldest_ts = df_chunk["timestamp"].min()
                    end_time = pd.to_datetime(oldest_ts).tz_convert("Asia/Seoul") - pd.Timedelta(milliseconds=1)
                    if len(df_chunk) < limit:
                        break
                except Exception as be:
                    print(f"[❌ Binance 수집 실패] {symbol}-{strategy} → {be}")
                    break

        df_binance = _normalize_df(pd.concat(df_binance, ignore_index=True)) if df_binance else pd.DataFrame()

        # 3) 병합 + 정리
        df_list = [df for df in [df_bybit, df_binance] if df is not None and not df.empty]
        df = _normalize_df(pd.concat(df_list, ignore_index=True)) if df_list else pd.DataFrame()
        df = _clip_tail(df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True), limit)

        total_count = len(df)
        # 🔧 최소 보장 수량
        min_required = max(60, int(limit * 0.90))
        if total_count < min_required:
            print(f"[⚠️ 수집 수량 부족] {symbol}-{strategy} → 총 {total_count}개 (최소보장 {min_required}, 목표 {limit}) → 통합 재시도")
            # 최종 통합 재시도 (Bybit+Binance 병행 수집기)
            df_retry = get_merged_kline_by_strategy(symbol, strategy)
            if not df_retry.empty and len(df_retry) > total_count:
                df = _clip_tail(df_retry, limit)
                total_count = len(df)

        if total_count < min_required:
            print(f"[🚨 최종 부족] {symbol}-{strategy} → {total_count}/{min_required} (학습/예측 영향 가능)")
        else:
            print(f"[✅ 수집 성공] {symbol}-{strategy} → 총 {total_count}개")

        # 🔧 변경: 항상 augment 플래그와 학습충족 여부를 attrs에 명시
        df.attrs["augment_needed"] = total_count < limit
        df.attrs["enough_for_training"] = total_count >= min_required

        CacheManager.set(cache_key, df)
        return df

    except Exception as e:
        print(f"[❌ 데이터 수집 실패] {symbol}-{strategy} → {e}")
        safe_failed_result(symbol, strategy, reason=str(e))
        return pd.DataFrame()

# =========================
# 프리패치
# =========================
def prefetch_symbol_groups(strategy: str):
    for group in SYMBOL_GROUPS:
        for sym in group:
            try:
                get_kline_by_strategy(sym, strategy)
            except Exception as e:
                print(f"[⚠️ prefetch 실패] {sym}-{strategy}: {e}")

# =========================
# 실시간 티커
# =========================
def get_realtime_prices():
    url = f"{BASE_URL}/v5/market/tickers"
    params = {"category": "linear"}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return {}
        tickers = data["result"]["list"]
        symset = set(get_ALL_SYMBOLS())
        return {item["symbol"]: float(item["lastPrice"]) for item in tickers if item["symbol"] in symset}
    except:
        return {}

# =========================
# 피처 생성 (+ 시장 레짐 태깅)
# =========================
_feature_cache = {}

def compute_features(symbol: str, df: pd.DataFrame, strategy: str, required_features: list = None, fallback_input_size: int = None) -> pd.DataFrame:
    from config import FEATURE_INPUT_SIZE
    import ta

    cache_key = f"{symbol}-{strategy}-features"
    cached_feat = CacheManager.get(cache_key, ttl_sec=600)
    if cached_feat is not None:
        print(f"[캐시 HIT] {cache_key}")
        return cached_feat

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
    base_cols = ["open", "high", "low", "close", "volume"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[["timestamp"] + base_cols]

    if len(df) < 20:
        print(f"[⚠️ 피처 실패] {symbol}-{strategy} → row 수 부족: {len(df)}")
        safe_failed_result(symbol, strategy, reason=f"row 부족 {len(df)}")
        return df  # 최소 반환

    try:
        # ---------- 기본 기술지표 ----------
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
        import ta as _ta  # 일부 배포에서 별칭 사용 호환
        df["adx"] = _ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
        df["cci"] = _ta.trend.cci(df["high"], df["low"], df["close"], window=20, fillna=True)
        df["mfi"] = _ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14, fillna=True)
        df["obv"] = _ta.volume.on_balance_volume(df["close"], df["volume"], fillna=True)
        df["atr"] = _ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14, fillna=True)
        df["williams_r"] = _ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14, fillna=True)
        df["stoch_k"] = _ta.momentum.stoch(df["high"], df["low"], df["close"], fillna=True)
        df["stoch_d"] = _ta.momentum.stoch_signal(df["high"], df["low"], df["close"], fillna=True)
        df["vwap"] = (df["volume"] * df["close"]).cumsum() / (df["volume"].cumsum() + 1e-6)

        # ---------- 시장 레짐 태깅 (옵션) ----------
        regime_cfg = get_REGIME()
        if regime_cfg.get("enabled", False):
            atr_win = int(regime_cfg.get("atr_window", 14))
            trend_win = int(regime_cfg.get("trend_window", 50))
            vol_high_pct = float(regime_cfg.get("vol_high_pct", 0.9))
            vol_low_pct = float(regime_cfg.get("vol_low_pct", 0.5))

            df["atr_val"] = _ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_win, fillna=True)
            thr_high = df["atr_val"].quantile(vol_high_pct)
            thr_low = df["atr_val"].quantile(vol_low_pct)
            df["vol_regime"] = np.where(df["atr_val"] >= thr_high, 2,
                                 np.where(df["atr_val"] <= thr_low, 0, 1))

            df["ma_trend"] = df["close"].rolling(window=trend_win, min_periods=1).mean()
            slope = df["ma_trend"].diff()
            df["trend_regime"] = np.where(slope > 0, 2, np.where(slope < 0, 0, 1))
            df["regime_tag"] = df["vol_regime"] * 3 + df["trend_regime"]

        # ---------- 정리/스케일 ----------
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        feature_cols = [c for c in df.columns if c != "timestamp"]
        from config import FEATURE_INPUT_SIZE as _FIS
        if len(feature_cols) < _FIS:
            for i in range(len(feature_cols), _FIS):
                pad_col = f"pad_{i}"
                df[pad_col] = 0.0
                feature_cols.append(pad_col)

        # downcast 후 스케일 → float32 유지
        df[feature_cols] = _downcast_numeric(df[feature_cols]).astype(np.float32)
        df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols])

    except Exception as e:
        print(f"[❌ compute_features 실패] feature 계산 예외 → {e}")
        safe_failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        return df  # 최소 구조라도 반환

    if df.empty or df.isnull().values.any():
        print(f"[❌ compute_features 실패] 결과 DataFrame 문제 → 빈 df 또는 NaN 존재")
        safe_failed_result(symbol, strategy, reason="최종 결과 DataFrame 오류")
        return df

    print(f"[✅ 완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    print(f"[🔍 feature 상태] {symbol}-{strategy} → shape: {df.shape}, NaN: {df.isnull().values.any()}, 컬럼수: {len(df.columns)}")
    CacheManager.set(cache_key, df)
    return df
