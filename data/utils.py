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
    "단기": {"interval": "240", "limit": 600},  # 4시간봉, 최대 1000개
    "중기": {"interval": "D", "limit": 600},    # 1일봉, 최대 1000개
    "장기": {"interval": "D", "limit": 600}     # 1일봉, 최대 1000개
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
    from config import get_NUM_CLASSES, MIN_FEATURES
    from logger import log_prediction
    NUM_CLASSES = get_NUM_CLASSES()
    from collections import Counter
    import random

    X, y = [], []

    def get_symbol_safe():
        if features and isinstance(features[0], dict) and "symbol" in features[0]:
            return features[0]["symbol"]
        return "UNKNOWN"

    if not features or len(features) <= window:
        print(f"[⚠️ 부족] features length={len(features) if features else 0}, window={window} → dummy 반환")
        dummy_X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        dummy_y = np.array([-1], dtype=np.int64)
        log_prediction(
            symbol=get_symbol_safe(),
            strategy=strategy,
            direction="dummy",
            entry_price=0,
            target_price=0,
            model="dummy_model",
            success=False,
            reason=f"window 부족 dummy (len={len(features) if features else 0}, window={window})",
            rate=0.0,
            return_value=0.0,
            volatility=False,
            source="create_dataset",
            predicted_class=-1,
            label=-1
        )
        return dummy_X, dummy_y

    df = pd.DataFrame(features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.drop(columns=["strategy"], errors="ignore")

    feature_cols = [c for c in df.columns if c not in ["timestamp"]]
    if not feature_cols:
        print("[⚠️ 부족] feature drop 결과 컬럼 없음 → dummy 반환")
        dummy_X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        dummy_y = np.array([-1], dtype=np.int64)
        log_prediction(symbol=get_symbol_safe(), strategy=strategy, direction="dummy", entry_price=0, target_price=0, model="dummy_model", success=False, reason="feature_cols 없음 dummy", rate=0.0, return_value=0.0, volatility=False, source="create_dataset", predicted_class=-1, label=-1)
        return dummy_X, dummy_y

    if len(feature_cols) < MIN_FEATURES:
        for i in range(len(feature_cols), MIN_FEATURES):
            pad_col = f"pad_{i}"
            df[pad_col] = 0.0
            feature_cols.append(pad_col)

    scaler = MinMaxScaler()
    try:
        scaled = scaler.fit_transform(df[feature_cols])
    except Exception as e:
        print(f"[❌ create_dataset 실패] scaler fit_transform 실패 → {e}")
        return None, None

    df_scaled = pd.DataFrame(scaled, columns=feature_cols)
    df_scaled["timestamp"] = df["timestamp"].values

    if input_size and len(feature_cols) < input_size:
        for i in range(len(feature_cols), input_size):
            pad_col = f"pad_{i}"
            df_scaled[pad_col] = 0.0

    features = df_scaled.to_dict(orient="records")

    strategy_minutes = {"단기": 240, "중기": 1440, "장기": 10080}
    lookahead_minutes = strategy_minutes.get(strategy, 1440)
    class_ranges = [(-1.0 + 2.0 * i / NUM_CLASSES, -1.0 + 2.0 * (i + 1) / NUM_CLASSES) for i in range(NUM_CLASSES)]

    for i in range(window, len(features)):
        try:
            seq = features[i - window:i]
            base = features[i]
            entry_time = pd.to_datetime(base.get("timestamp"), errors="coerce")
            entry_price = float(base.get("close", 0.0))
            if pd.isnull(entry_time) or entry_price <= 0:
                continue

            future = [f for f in features[i + 1:] if "timestamp" in f and pd.to_datetime(f["timestamp"], errors="coerce") - entry_time <= pd.Timedelta(minutes=lookahead_minutes)]
            if len(seq) != window or len(future) < 1:
                continue

            max_future_price = max(f.get("high", f.get("close", entry_price)) for f in future)
            gain = float((max_future_price - entry_price) / (entry_price + 1e-6))
            gain = max(-1.0, min(1.0, gain))  # ✅ gain 범위 제한

            cls = next((j for j, (low, high) in enumerate(class_ranges) if low <= gain <= high), NUM_CLASSES-1)
            sample = [[float(r.get(c, 0.0)) for c in feature_cols] for r in seq]
            if input_size:
                for j in range(len(sample)):
                    row = sample[j]
                    if len(row) < input_size:
                        row.extend([0.0] * (input_size - len(row)))
                    elif len(row) > input_size:
                        sample[j] = row[:input_size]
            X.append(sample)
            y.append(cls)
        except Exception as e:
            print(f"[예외] {e} → i={i}")
            continue

    # ✅ 샘플은 있으나 라벨이 생성되지 않은 경우 → 중립 라벨 부여
    if len(y) == 0:
        if len(X) > 0:
            print(f"[⚠️ 라벨 없음] → 중립 라벨 부여 후 진행 (샘플 수: {len(X)})")
            y = [NUM_CLASSES // 2] * len(X)
        else:
            print(f"[❌ create_dataset 실패] 샘플 생성 실패 → 유효 샘플 없음")
            dummy_X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
            dummy_y = np.array([-1], dtype=np.int64)
            log_prediction(
                symbol=get_symbol_safe(),
                strategy=strategy,
                direction="dummy",
                entry_price=0,
                target_price=0,
                model="dummy_model",
                success=False,
                reason="샘플 생성 실패 (유효 라벨 없음)",
                rate=0.0,
                return_value=0.0,
                volatility=False,
                source="create_dataset",
                predicted_class=-1,
                label=-1
            )
            return dummy_X, dummy_y

    # ✅ 최소 샘플수 10개 이상 보장
    if len(y) < 10:
        print(f"[⚠️ 부족] 샘플 수 {len(y)}개 → 최소 10개로 복제 보장")
        while len(y) < 10:
            X.append(X[0])
            y.append(y[0])

    counts = Counter(y)
    max_count = max(counts.values())
    for cls_id in range(NUM_CLASSES):
        cls_samples = [i for i, label in enumerate(y) if label == cls_id]
        needed = int(max_count * 0.8) - len(cls_samples)
        if cls_samples and needed > 0:
            for _ in range(needed):
                idx = random.choice(cls_samples)
                X.append(X[idx])
                y.append(cls_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"[✅ create_dataset 완료] 샘플 수: {len(y)}, 클래스 분포: {Counter(y)}")
    return X, y


# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}
_kline_cache_ttl = {}  # ✅ TTL 추가

import time

def get_kline_by_strategy(symbol: str, strategy: str):
    from predict import failed_result
    import os

    cache_key = f"{symbol}-{strategy}"
    cached_df = CacheManager.get(cache_key, ttl_sec=600)
    if cached_df is not None:
        return cached_df

    config = STRATEGY_CONFIG.get(strategy)
    if config is None:
        print(f"[❌ 실패] {symbol}-{strategy}: 전략 설정 없음")
        failed_result(symbol, strategy, reason="전략 설정 없음")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = get_kline(symbol, interval=config["interval"], limit=config["limit"])
    if df is None or not isinstance(df, pd.DataFrame):
        print(f"[❌ 실패] {symbol}-{strategy}: get_kline() → None 반환 또는 형식 오류")
        failed_result(symbol, strategy, reason="get_kline 반환 오류")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    required_cols = ["open", "high", "low", "close", "volume", "timestamp"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if col != "timestamp" else pd.Timestamp.now()

    df = df[required_cols]

    if len(df) < 5:
        failed_result(symbol, strategy, reason="row 부족")
        return df

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
                print(f"[경고] get_kline() → 필수 필드 누락: {symbol}, 재시도 {attempt+1}/{max_retry}")
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

            if "high" not in df.columns or df["high"].isnull().all() or (df["high"] == 0).all():
                print(f"[치명] get_kline() → 'high' 값 전부 비정상: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["datetime"] = df["timestamp"]

            return df

        except Exception as e:
            print(f"[에러] get_kline({symbol}) 실패 → {e}, 재시도 {attempt+1}/{max_retry}")
            time.sleep(1)

    print(f"[❌ 실패] get_kline() 최종 실패: {symbol}")
    return None


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

    if df is None or df.empty:
        print(f"[❌ compute_features 실패] 입력 DataFrame empty")
        failed_result(symbol, strategy, reason="입력DataFrame empty")
        return None

    df = df.copy()

    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now")

    df["strategy"] = strategy

    try:
        base_cols = ["open", "high", "low", "close", "volume"]
        df = df[["timestamp", "strategy"] + base_cols]

        # ✅ 기존 feature engineering 유지
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

        # ✅ 고급 기술적 지표 추가
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
        df["vwap"] = (df["volume"] * df["close"]).cumsum() / df["volume"].cumsum()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        feature_cols = [c for c in df.columns if c not in ["timestamp", "strategy"]]
        print(f"[info] compute_features 생성 feature 개수: {len(feature_cols)} → {feature_cols}")

        if len(feature_cols) < FEATURE_INPUT_SIZE:
            pad_cols = []
            for i in range(len(feature_cols), FEATURE_INPUT_SIZE):
                pad_col = f"pad_{i}"
                df[pad_col] = 0.0
                pad_cols.append(pad_col)
            feature_cols += pad_cols
            print(f"[info] feature padding 적용: {pad_cols}")

        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    except Exception as e:
        print(f"[❌ compute_features 실패] feature 계산 예외 → {e}")
        failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        return None

    required_cols = ["timestamp", "close", "high"]
    missing_cols = [col for col in required_cols if col not in df.columns or df[col].isnull().any()]
    if missing_cols or df.empty:
        print(f"[❌ compute_features 실패] 필수 컬럼 누락 또는 NaN 존재: {missing_cols}, rows={len(df)}")
        failed_result(symbol, strategy, reason=f"필수컬럼누락 또는 NaN: {missing_cols}")
        return None

    if len(df) < 5:
        print(f"[❌ compute_features 실패] 데이터 row 부족 ({len(df)} rows)")
        failed_result(symbol, strategy, reason="row 부족")
        return None

    print(f"[✅ 완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    CacheManager.set(cache_key, df)
    return df


# data/utils.py 맨 아래에 추가

SYMBOL_GROUPS = [SYMBOLS[i:i+5] for i in range(0, len(SYMBOLS), 5)]
