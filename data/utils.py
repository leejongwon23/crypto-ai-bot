import requests
import pandas as pd
import numpy as np
import time
import pytz

BASE_URL = "https://api.bybit.com"
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}

SYMBOLS = [  # 사용되는 모든 심볼
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
    "단기": {"interval": "240", "limit": 300},
    "중기": {"interval": "D", "limit": 300},
    "장기": {"interval": "w", "limit": 300}
}

DEFAULT_MIN_GAIN = {
    "단기": 0.01,
    "중기": 0.03,
    "장기": 0.05
}

def get_min_gain(symbol: str, strategy: str):
    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < 20:
            return DEFAULT_MIN_GAIN.get(strategy, 0.03)
        vol = df["close"].pct_change().rolling(20).std().iloc[-1]
        return max(round(vol * 1.2, 4), DEFAULT_MIN_GAIN.get(strategy, 0.01))
    except Exception:
        return DEFAULT_MIN_GAIN.get(strategy, 0.01)

def get_btc_dominance():
    global BTC_DOMINANCE_CACHE
    now = time.time()
    if now - BTC_DOMINANCE_CACHE["timestamp"] < 1800:
        return BTC_DOMINANCE_CACHE["value"]
    try:
        res = requests.get("https://api.coinpaprika.com/v1/global", timeout=10)
        res.raise_for_status()
        data = res.json()
        dom = float(data["bitcoin_dominance_percentage"]) / 100
        BTC_DOMINANCE_CACHE = {"value": round(dom, 4), "timestamp": now}
        return BTC_DOMINANCE_CACHE["value"]
    except Exception as e:
        print(f"[ERROR] BTC 도미넌스 조회 실패: {e}")
        return BTC_DOMINANCE_CACHE["value"]

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
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", *_]  # extra columns 무시
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.astype({
            "timestamp": "int64", "open": "float", "high": "float",
            "low": "float", "close": "float", "volume": "float"
        })
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] {symbol} 데이터 요청 실패: {e}")
        return None

def get_kline_by_strategy(symbol: str, strategy: str):
    config = STRATEGY_CONFIG.get(strategy)
    if config is None:
        raise ValueError(f"전략 '{strategy}'에 대한 설정 없음")
    return get_kline(symbol, interval=config["interval"], limit=config["limit"])

def get_realtime_prices():
    url = f"{BASE_URL}/v5/market/tickers"
    params = {"category": "linear"}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return {}
        return {item["symbol"]: float(item["lastPrice"]) for item in data["result"]["list"] if item["symbol"] in SYMBOLS}
    except Exception as e:
        print(f"[ERROR] 실시간 가격 조회 실패: {e}")
        return {}

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    bb_ma = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bollinger'] = (df['close'] - bb_ma) / (2 * bb_std + 1e-6)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['trend_slope'] = df['ema10'].diff()
    df['percent_diff'] = (df['close'] - df['ma20']) / (df['ma20'] + 1e-6)
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
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-6)
    df['cci'] = cci

    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-6)

    df["btc_dominance"] = get_btc_dominance()

    df = df.dropna()
    return df[[
        'close', 'volume', 'ma5', 'ma20', 'rsi', 'macd',
        'bollinger', 'volatility', 'trend_slope',
        'percent_diff', 'volume_delta',
        'obv', 'cci', 'stoch_rsi', 'btc_dominance'
    ]]
