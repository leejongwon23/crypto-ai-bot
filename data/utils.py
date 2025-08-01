# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}

import requests
import pandas as pd
import numpy as np
import time
import pytz
from sklearn.preprocessing import MinMaxScaler


BASE_URL = "https://api.bybit.com"
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}

SYMBOLS = [
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

STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 1000},   # 4시간봉 (240분)
    "중기": {"interval": "D",   "limit": 500},    # 1일봉
    "장기": {"interval": "2D",  "limit": 300}     # 2일봉
}

# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
import time

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
            del cls._ttl[key]
            print(f"[캐시 DELETE] {key}")

    @classmethod
    def clear(cls):
        cls._cache.clear()
        cls._ttl.clear()
        print("[캐시 CLEAR ALL]")


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

import numpy as np

def create_dataset(features, window=10, strategy="단기", input_size=None):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from config import MIN_FEATURES
    from logger import log_prediction
    from collections import Counter
    import random

    X, y = [], []

    def get_symbol_safe():
        if isinstance(features, list) and features and isinstance(features[0], dict) and "symbol" in features[0]:
            return features[0]["symbol"]
        return "UNKNOWN"

    if not isinstance(features, list) or len(features) <= window:
        print(f"[⚠️ 부족] features length={len(features) if isinstance(features, list) else 'Invalid'}, window={window} → dummy 반환")
        dummy_X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        dummy_y = np.array([-1], dtype=np.int64)
        log_prediction(symbol=get_symbol_safe(), strategy=strategy, direction="dummy", entry_price=0,
                       target_price=0, model="dummy_model", success=False, reason="입력 feature 부족",
                       rate=0.0, return_value=0.0, volatility=False, source="create_dataset",
                       predicted_class=-1, label=-1)
        return dummy_X, dummy_y

    try:
        df = pd.DataFrame(features)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.drop(columns=["strategy"], errors="ignore")

        feature_cols = [c for c in df.columns if c not in ["timestamp"]]
        if not feature_cols:
            raise ValueError("feature_cols 없음")

        if len(feature_cols) < MIN_FEATURES:
            for i in range(len(feature_cols), MIN_FEATURES):
                pad_col = f"pad_{i}"
                df[pad_col] = 0.0
                feature_cols.append(pad_col)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols])
        df_scaled = pd.DataFrame(scaled, columns=feature_cols)
        df_scaled["timestamp"] = df["timestamp"].values
        df_scaled["high"] = df["high"] if "high" in df.columns else df["close"]

        if input_size and len(feature_cols) < input_size:
            for i in range(len(feature_cols), input_size):
                pad_col = f"pad_{i}"
                df_scaled[pad_col] = 0.0

        features = df_scaled.to_dict(orient="records")

        strategy_minutes = {"단기": 240, "중기": 1440, "장기": 2880}
        lookahead_minutes = strategy_minutes.get(strategy, 1440)

        valid_gains = []
        samples = []

        for i in range(window, len(features)):
            try:
                seq = features[i - window:i]
                base = features[i]
                entry_time = pd.to_datetime(base.get("timestamp"), errors="coerce")
                entry_price = float(base.get("close", 0.0))
                if pd.isnull(entry_time) or entry_price <= 0:
                    continue

                future = [f for f in features[i + 1:] if pd.to_datetime(f.get("timestamp", None)) - entry_time <= pd.Timedelta(minutes=lookahead_minutes)]
                valid_prices = [f.get("high", f.get("close", entry_price)) for f in future if f.get("high", 0) > 0]
                if len(seq) != window or not valid_prices:
                    continue

                max_future_price = max(valid_prices)
                gain = float((max_future_price - entry_price) / (entry_price + 1e-6))
                valid_gains.append(gain)

                sample = [[float(r.get(c, 0.0)) for c in feature_cols] for r in seq]
                if input_size:
                    for j in range(len(sample)):
                        row = sample[j]
                        if len(row) < input_size:
                            row.extend([0.0] * (input_size - len(row)))
                        elif len(row) > input_size:
                            sample[j] = row[:input_size]

                samples.append((sample, gain))
            except Exception:
                continue

        if not samples or not valid_gains:
            print("[❌ 수익률 없음] dummy 반환")
            dummy_X = np.random.normal(0, 1, size=(10, window, input_size if input_size else MIN_FEATURES)).astype(np.float32)
            dummy_y = np.random.randint(0, 5, size=(10,))  # 최소 클래스 수 가정
            return dummy_X, dummy_y

        # ✅ 클래스 수 동적 계산 (최대 21개, 최소 3개)
        min_gain, max_gain = min(valid_gains), max(valid_gains)
        spread = max_gain - min_gain
        est_class = int(spread / 0.01)  # 1% 단위로 나눈 것 기준
        num_classes = max(3, min(21, est_class))

        step = spread / num_classes if num_classes > 0 else 1e-6
        if step == 0:
            step = 1e-6

        for sample, gain in samples:
            cls = min(int((gain - min_gain) / step), num_classes - 1)
            X.append(sample)
            y.append(cls)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        print(f"[✅ create_dataset 완료] 샘플 수: {len(y)}, X.shape={X.shape}, 동적 클래스 수: {num_classes}, 분포: {Counter(y)}")
        return X, y, num_classes  # ✅ 클래스 수 함께 반환

    except Exception as e:
        print(f"[❌ 최상위 예외] create_dataset 실패 → {e}")
        dummy_X = np.random.normal(0, 1, size=(10, window, input_size if input_size else MIN_FEATURES)).astype(np.float32)
        dummy_y = np.random.randint(0, 5, size=(10,))
        return dummy_X, dummy_y, 5

# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}
_kline_cache_ttl = {}  # ✅ TTL 추가

import time

def get_kline_by_strategy(symbol: str, strategy: str):
    from predict import failed_result
    import os, time
    import pandas as pd

    cache_key = f"{symbol}-{strategy}"
    cached_df = CacheManager.get(cache_key, ttl_sec=600)
    if cached_df is not None:
        return cached_df

    config = STRATEGY_CONFIG.get(strategy)
    if config is None:
        print(f"[❌ 실패] {symbol}-{strategy}: 전략 설정 없음")
        failed_result(symbol, strategy, reason="전략 설정 없음")
        return None

    required_rows = config.get("limit", 100)
    df = None
    last_valid_df = None

    # ✅ 재시도 횟수 확장 (3 → 5)
    for attempt in range(5):
        try:
            df = get_kline(symbol, interval=config["interval"], limit=config["limit"])
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                row_count = len(df)
                if row_count >= required_rows:
                    last_valid_df = df.copy()
                    print(f"[✅ get_kline 성공] {symbol}-{strategy} row={row_count}")
                    break
                else:
                    print(f"[⚠️ get_kline 시도 {attempt+1}/5] row 부족: {row_count} / 필요: {required_rows}")
                    last_valid_df = df.copy()
            else:
                print(f"[⚠️ get_kline 시도 {attempt+1}/5] 빈 데이터 또는 DataFrame 아님")
        except Exception as e:
            print(f"[⚠️ get_kline 예외 - 시도 {attempt+1}/5] {symbol}-{strategy} → {e}")

        time.sleep(1 + attempt)  # 재시도 간 대기 시간 점진 증가

    # ✅ 최종 데이터 확보 여부 판단
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[❌ 실패] get_kline() 최종 실패: {symbol}-{strategy}, 데이터 없음")
        failed_result(symbol, strategy, reason="캔들 데이터 없음")
        return None

    # 마지막 유효 데이터라도 있으면 사용
    if len(df) < required_rows:
        if last_valid_df is not None and not last_valid_df.empty:
            df = last_valid_df
        print(f"[⚠️ 데이터 부족 → 증강 예정] {symbol}-{strategy} row={len(df)} / 필요: {required_rows}")

    # ✅ 필수 컬럼 보정
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if col != "timestamp" else pd.Timestamp.now()

    df = df[required_cols]

    # ✅ 증강 필요 플래그 추가
    df.attrs["augment_needed"] = len(df) < required_rows

    # 캐시에 저장
    CacheManager.set(cache_key, df)
    return df


# ✅ SYMBOL_GROUPS batch prefetch 함수 추가

def prefetch_symbol_groups(strategy: str):
    for group in SYMBOL_GROUPS:
        for symbol in group:
            try:
                get_kline_by_strategy(symbol, strategy)
            except Exception as e:
                print(f"[⚠️ prefetch 실패] {symbol}-{strategy}: {e}")

def get_kline(symbol: str, interval: str = "60", limit: int = 300, max_retry: int = 3) -> pd.DataFrame:
    import time

    for attempt in range(max_retry):
        try:
            url = f"{BASE_URL}/v5/market/kline"
            params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()

            if "result" not in data or "list" not in data["result"]:
                print(f"[경고] get_kline() → 데이터 응답 구조 이상: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            raw = data["result"]["list"]
            if not raw or len(raw[0]) < 6:
                print(f"[경고] get_kline() → 필수 필드 누락 또는 빈 응답: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")

            essential = ["open", "high", "low", "close", "volume"]
            df.dropna(subset=essential, inplace=True)
            if df.empty:
                print(f"[경고] get_kline() → 필수값 결측: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            if df["high"].isnull().all() or (df["high"] == 0).all():
                print(f"[치명] get_kline() → 'high' 값 이상치만 존재: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["datetime"] = df["timestamp"]

            if len(df) < 10:
                print(f"[⚠️ 경고] {symbol}-{interval} → 캔들 수 부족 ({len(df)} rows)")

            print(f"[✅ get_kline 완료] {symbol}-{interval} → {len(df)}개 캔들 로드됨")  # 🔍 추가된 로그
            return df

        except Exception as e:
            print(f"[에러] get_kline({symbol}) 실패 → {e}, 재시도 {attempt+1}/{max_retry}")
            time.sleep(1)

    print(f"[❌ 실패] get_kline() 최종 실패: {symbol}")
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])


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
        return {item["symbol"]: float(item["lastPrice"]) for item in tickers if item["symbol"] in SYMBOLS}
    except:
        return {}


_feature_cache = {}

def compute_features(symbol: str, df: pd.DataFrame, strategy: str, required_features: list = None, fallback_input_size: int = None) -> pd.DataFrame:
    from predict import failed_result
    from config import FEATURE_INPUT_SIZE
    import ta
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    cache_key = f"{symbol}-{strategy}-features"
    cached_feat = CacheManager.get(cache_key, ttl_sec=600)
    if cached_feat is not None:
        print(f"[캐시 HIT] {cache_key}")
        return cached_feat

    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        print(f"[❌ compute_features 실패] 입력 DataFrame empty or invalid")
        failed_result(symbol, strategy, reason="입력DataFrame empty")
        return pd.DataFrame()

    df = df.copy()
    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now")

    df["strategy"] = strategy  # 로그용으로만 사용
    base_cols = ["open", "high", "low", "close", "volume"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = 0.0

    # ✅ 전략명 제거 (숫자 벡터 오류 방지)
    df = df[["timestamp"] + base_cols]

    if len(df) < 20:
        print(f"[⚠️ 피처 실패] {symbol}-{strategy} → row 수 부족: {len(df)}")
        failed_result(symbol, strategy, reason=f"row 부족 {len(df)}")
        return pd.DataFrame()

    try:
        df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
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
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
        df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20, fillna=True)
        df["mfi"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14, fillna=True)
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"], fillna=True)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14, fillna=True)
        df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14, fillna=True)
        df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"], fillna=True)
        df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], fillna=True)
        df["vwap"] = (df["volume"] * df["close"]).cumsum() / (df["volume"].cumsum() + 1e-6)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        feature_cols = [c for c in df.columns if c != "timestamp"]
        if len(feature_cols) < FEATURE_INPUT_SIZE:
            for i in range(len(feature_cols), FEATURE_INPUT_SIZE):
                pad_col = f"pad_{i}"
                df[pad_col] = 0.0
                feature_cols.append(pad_col)

        df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols])

    except Exception as e:
        print(f"[❌ compute_features 실패] feature 계산 예외 → {e}")
        failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        return pd.DataFrame()

    if df.empty or df.isnull().values.any():
        print(f"[❌ compute_features 실패] 결과 DataFrame 문제 → 빈 df 또는 NaN 존재")
        failed_result(symbol, strategy, reason="최종 결과 DataFrame 오류")
        return pd.DataFrame()

    print(f"[✅ 완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    print(f"[🔍 feature 상태] {symbol}-{strategy} → shape: {df.shape}, NaN: {df.isnull().values.any()}, 컬럼수: {len(df.columns)}")
    CacheManager.set(cache_key, df)
    return df


# data/utils.py 맨 아래에 추가

SYMBOL_GROUPS = [SYMBOLS[i:i+5] for i in range(0, len(SYMBOLS), 5)]
