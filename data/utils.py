# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}

import requests
import pandas as pd
import numpy as np
import time
import pytz

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

def create_dataset(features, window=20, strategy="단기"):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    X, y = [], []

    if not features or len(features) <= window:
        print(f"[❌ 스킵] features 부족 → len={len(features) if features else 0}")
        return np.array([]), np.array([])

    try:
        columns = [c for c in features[0].keys() if c != "timestamp"]
    except Exception as e:
        print(f"[오류] features[0] 키 확인 실패 → {e}")
        return np.array([]), np.array([])

    required_keys = {"timestamp", "close", "high"}
    if not all(all(k in f for k in required_keys) for f in features):
        print("[❌ 스킵] 필수 키 누락된 feature 존재")
        return np.array([]), np.array([])

    df = pd.DataFrame(features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close", "high"]).sort_values("timestamp").reset_index(drop=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.drop(columns=["timestamp"]))
    df_scaled = pd.DataFrame(scaled, columns=columns)
    df_scaled["timestamp"] = df["timestamp"].values

    features = df_scaled.to_dict(orient="records")

    # ✅ 수정된 class_ranges – 하락 구간 포함
    class_ranges = [
        (-1.00, -0.50), (-0.50, -0.30), (-0.30, -0.10),
        (-0.10, -0.05), (-0.05, -0.01), (-0.01, 0.00),
        (0.00, 0.01), (0.01, 0.05), (0.05, 0.10),
        (0.10, 0.20), (0.20, 0.40), (0.40, 0.70),
        (0.70, 1.00), (1.00, 1.50), (1.50, 2.00), (2.00, 5.00)
    ]

    strategy_minutes = {"단기": 240, "중기": 1440, "장기": 10080}
    lookahead_minutes = strategy_minutes.get(strategy, 1440)

    for i in range(window, len(features) - 3):
        try:
            seq = features[i - window:i]
            base = features[i]
            entry_time = pd.to_datetime(base.get("timestamp"), errors="coerce")
            entry_price = float(base.get("close", 0.0))

            if pd.isnull(entry_time):
                continue

            future = [
                f for f in features[i + 1:]
                if "timestamp" in f and pd.to_datetime(f["timestamp"], errors="coerce") - entry_time <= pd.Timedelta(minutes=lookahead_minutes)
            ]

            if len(seq) != window or len(future) < 1:
                continue

            max_future_price = max(f.get("high", f.get("close", entry_price)) for f in future)
            gain = (max_future_price - entry_price) / (entry_price + 1e-6)

            # ✅ gain debug print 추가
            print(f"[gain debug] entry_price: {entry_price}, max_future_price: {max_future_price}, gain: {gain}")

            cls = next((j for j, (low, high) in enumerate(class_ranges) if low <= gain < high), -1)
            if cls == -1:
                continue

            sample = [[float(r.get(c, 0.0)) for c in columns] for r in seq]
            if any(len(row) != len(columns) for row in sample):
                continue

            X.append(sample)
            y.append(cls)

        except Exception as e:
            print(f"[예외 발생] ❌ {e} → i={i}")
            continue

    if y:
        labels, counts = np.unique(y, return_counts=True)
        print(f"[📊 클래스 분포] → {dict(zip(labels, counts))}")
    else:
        print("[⚠️ 경고] 생성된 라벨 없음")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def get_kline_by_strategy(symbol: str, strategy: str):
    global _kline_cache
    cache_key = f"{symbol}-{strategy}"
    if cache_key in _kline_cache:
        print(f"[캐시 사용] {cache_key}")
        return _kline_cache[cache_key]

    config = STRATEGY_CONFIG.get(strategy)
    if config is None:
        print(f"[오류] 전략 설정 없음: {strategy}")
        return None

    df = get_kline(symbol, interval=config["interval"], limit=config["limit"])
    if df is None or not isinstance(df, pd.DataFrame):
        print(f"[❌ 실패] {symbol}-{strategy}: get_kline() → None 또는 형식 오류")
        return None

    required_cols = ["open", "high", "low", "close", "volume", "timestamp"]
    missing_or_nan = [col for col in required_cols if col not in df.columns or df[col].isnull().any()]
    if missing_or_nan:
        print(f"[❌ 실패] {symbol}-{strategy}: 필수 컬럼 누락 또는 NaN 존재: {missing_or_nan}")
        return None

    print(f"[확인] {symbol}-{strategy}: 데이터 {len(df)}개 확보")
    _kline_cache[cache_key] = df
    return df


def get_kline(symbol: str, interval: str = "60", limit: int = 300) -> pd.DataFrame:
    try:
        url = f"{BASE_URL}/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        if "result" not in data or "list" not in data["result"]:
            print(f"[경고] get_kline() → 데이터 응답 구조 이상: {symbol}")
            return None

        raw = data["result"]["list"]
        if not raw or len(raw[0]) < 6:
            print(f"[경고] get_kline() → 필수 필드 누락: {symbol}")
            return None

        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")

        # ✅ 필수 컬럼 누락 or 전부 NaN or 전부 0인 경우 제거
        essential = ["open", "high", "low", "close", "volume"]
        df.dropna(subset=essential, inplace=True)
        if df.empty:
            print(f"[경고] get_kline() → 필수값 결측: {symbol}")
            return None

        if "high" not in df.columns or df["high"].isnull().all() or (df["high"] == 0).all():
            print(f"[치명] get_kline() → 'high' 값 전부 비정상: {symbol}")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["datetime"] = df["timestamp"]

        return df

    except Exception as e:
        print(f"[에러] get_kline({symbol}) 실패 → {e}")
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

def compute_features(symbol: str, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    global _feature_cache
    cache_key = f"{symbol}-{strategy}"
    if cache_key in _feature_cache:
        print(f"[캐시 사용] {cache_key} 피처")
        return _feature_cache[cache_key]

    df = df.copy()

    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now")

    if "high" not in df.columns or df["high"].isnull().all():
        print(f"[⚠️ 대체] {symbol}-{strategy} → 'high' 컬럼 누락 → 'close'로 대체")
        df["high"] = df["close"]

    if "low" not in df.columns or df["low"].isnull().all():
        df["low"] = df["close"]
    if "open" not in df.columns or df["open"].isnull().all():
        df["open"] = df["close"]

    df['ma20'] = df['close'].rolling(window=20).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))

    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow

    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['trend_score'] = df['close'].pct_change(periods=3)

    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-6)

    tp = (df['high'] + df['low'] + df['close']) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    df['cci'] = cci

    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    bb_ma = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bollinger'] = (df['close'] - bb_ma) / (2 * bb_std + 1e-6)

    if strategy == "중기":
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_cross'] = df['ema5'] - df['ema20']
    elif strategy == "장기":
        df['volume_cumsum'] = df['volume'].cumsum()
        df['roc'] = df['close'].pct_change(periods=12)
        mf = df["close"] * df["volume"]
        pos_mf = mf.where(df["close"] > df["close"].shift(), 0)
        neg_mf = mf.where(df["close"] < df["close"].shift(), 0)
        mf_ratio = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-6)
        df["mfi"] = 100 - (100 / (1 + mf_ratio))

    base = [
        "timestamp", "open", "high", "low", "close", "volume",
        "ma20", "rsi", "macd", "bollinger", "volatility",
        "trend_score", "stoch_rsi", "cci", "obv"
    ]
    mid_extra = ["ema_cross"]
    long_extra = ["volume_cumsum", "roc", "mfi"]

    if strategy == "중기":
        base += mid_extra
    elif strategy == "장기":
        base += long_extra

    df = df[base].dropna().reset_index(drop=True)

    required_cols = ["timestamp", "close", "high"]
    missing_cols = [col for col in required_cols if col not in df.columns or df[col].isnull().any()]
    if missing_cols or df.empty:
        print(f"[❌ compute_features 실패] 필수 컬럼 누락 또는 NaN 존재: {missing_cols}")
        return None

    print(f"[완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    _feature_cache[cache_key] = df
    return df

# data/utils.py 맨 아래에 추가

SYMBOL_GROUPS = [SYMBOLS[i:i+5] for i in range(0, len(SYMBOLS), 5)]
