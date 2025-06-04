# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마

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
    "단기": {"interval": "60", "limit": 300},    # 1시간봉
    "중기": {"interval": "240", "limit": 300},  # 4시간봉
    "장기": {"interval": "D", "limit": 300}     # 1일봉
}

DEFAULT_MIN_GAIN = {
    "단기": 0.01,
    "중기": 0.03,
    "장기": 0.05
}

def get_min_gain(symbol: str, strategy: str):
    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < 10:
            return DEFAULT_MIN_GAIN[strategy]
        df = df.tail(7)
        pct_changes = df["close"].pct_change().abs()
        avg_volatility = pct_changes.mean()
        return max(round(avg_volatility, 4), DEFAULT_MIN_GAIN[strategy])
    except:
        return DEFAULT_MIN_GAIN[strategy]

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

def create_dataset(features, window=20, strategy="단기"):
    X, y = [], []
    if not features or len(features) <= window:
        return X, y

    # ✅ 명시적 컬럼 순서 정의
    columns = [c for c in features[0].keys() if c != "timestamp"]

    for i in range(window, len(features)):
        try:
            seq = features[i - window:i]
            if len(seq) != window:
                continue
            x = [[row.get(c, 0.0) for c in columns] for row in seq]

            future = features[i:]
            direction = None
            entry_price = features[i]["close"]
            gain = 0

            if strategy == "단기":
                lookahead = min(8, len(future))
            elif strategy == "중기":
                lookahead = min(24, len(future))
            elif strategy == "장기":
                lookahead = min(48, len(future))
            else:
                lookahead = min(12, len(future))

            highs = [f.get("high", entry_price) for f in future[:lookahead]]
            lows = [f.get("low", entry_price) for f in future[:lookahead]]

            max_gain = max((h - entry_price) / entry_price for h in highs)
            max_loss = max((entry_price - l) / entry_price for l in lows)

            if max_gain > max_loss:
                gain = max_gain
                direction = "롱"
            else:
                gain = -max_loss
                direction = "숏"

            # ✅ 수익률 → 클래스 (16개 구간)
            bins = [-0.10, -0.07, -0.05, -0.03, -0.015, -0.005,
                     0.005, 0.015, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
            cls = next((i for i, b in enumerate(bins) if gain < b), len(bins)-1)

            if cls < 0 or cls >= 16:
                continue

            X.append(x)
            y.append(cls)
        except:
            continue

    return np.array(X), np.array(y)


def get_kline(symbol: str, interval: str = "60", limit: int = 200):
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return None
        rows = data["result"]["list"]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.astype({
            "timestamp": "int64", "open": "float", "high": "float",
            "low": "float", "close": "float", "volume": "float"
        })
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except:
        return None

def get_kline_by_strategy(symbol: str, strategy: str) -> pd.DataFrame:
    from_binance = __import__("data.binance").binance.get_klines
    interval_map = {"단기": "15m", "중기": "1h", "장기": "4h"}
    lookback_map = {"단기": 1000, "중기": 1000, "장기": 1000}
    
    interval = interval_map[strategy]
    lookback = lookback_map[strategy]

    df = from_binance(symbol, interval, lookback)

    if df is None or len(df) < 100:
        print(f"[경고] {symbol}-{strategy}: get_kline_by_strategy() 데이터 수 부족 → {len(df) if df is not None else 'None'}")
    else:
        print(f"[확인] {symbol}-{strategy}: 데이터 수량 → {len(df)}")

    return df


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

def compute_features(symbol: str, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = df.copy()

    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]

    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    bb_ma = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bollinger'] = (df['close'] - bb_ma) / (2 * bb_std)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['trend_score'] = df['ema10'].pct_change()
    df['current_vs_ma20'] = (df['close'] / (df['ma20'] + 1e-6)) - 1
    df['volume_delta'] = df['volume'].diff()

    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    tp = (df['high'] + df['low'] + df['close']) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    df['cci'] = cci

    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-6)

    btc_dom = get_btc_dominance()
    df['btc_dominance'] = btc_dom
    df['btc_dominance_diff'] = btc_dom - df['btc_dominance'].rolling(3).mean()

    # ✅ 추가 고급 피처
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = (df['close'] - df['open']).abs()
    df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-6)
    df['candle_upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['candle_range'] + 1e-6)
    df['candle_lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['candle_range'] + 1e-6)
    df['range_ratio'] = df['candle_range'] / (df['close'] + 1e-6)
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)

    if strategy == "중기":
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        tr = (df['high'] - df['low']).rolling(14).mean()
        dx = (abs(plus_dm - minus_dm) / (tr + 1e-6)) * 100
        df['adx'] = dx.rolling(14).mean()
        highest = df['high'].rolling(14).max()
        lowest = df['low'].rolling(14).min()
        df['willr'] = (highest - df['close']) / (highest - lowest + 1e-6) * -100

    if strategy == "장기":
        mf = df["close"] * df["volume"]
        pos_mf = mf.where(df["close"] > df["close"].shift(), 0)
        neg_mf = mf.where(df["close"] < df["close"].shift(), 0)
        mf_ratio = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-6)
        df["mfi"] = 100 - (100 / (1 + mf_ratio))
        df["roc"] = df["close"].pct_change(periods=12)

    base = [
        "timestamp", "close", "volume", "ma5", "ma20", "rsi", "macd", "bollinger", "volatility",
        "trend_score", "current_vs_ma20", "volume_delta", "obv", "cci", "stoch_rsi",
        "btc_dominance", "btc_dominance_diff",
        "candle_body_ratio", "candle_upper_shadow_ratio", "candle_lower_shadow_ratio",
        "range_ratio", "volume_ratio"
    ]
    mid_only = ["adx", "willr"]
    long_only = ["mfi", "roc"]

    extra = []
    if strategy == "중기":
        extra = mid_only
    elif strategy == "장기":
        extra = long_only

    df = df[base + extra]
    return df.dropna()

