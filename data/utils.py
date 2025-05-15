import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.bybit.com"
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}

SYMBOLS = [  # 전체 60개 심볼
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

def get_btc_dominance():
    global BTC_DOMINANCE_CACHE
    now = time.time()
    if now - BTC_DOMINANCE_CACHE["timestamp"] < 1800:  # 30분 캐시
        return BTC_DOMINANCE_CACHE["value"]

    try:
        url = "https://api.coinpaprika.com/v1/global"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        dom = data["market_cap_percentage"]["btc"]
        value = round(dom / 100, 4)
        BTC_DOMINANCE_CACHE = {"value": value, "timestamp": now}
        return value
    except Exception as e:
        print(f"[ERROR] BTC 도미넌스 조회 실패 (CoinPaprika): {e}")
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
        print(f"[DEBUG] {symbol} 요청 URL: {res.url}")
        preview = res.text[:300].replace("\n", "")
        print(f"[DEBUG] {symbol} 응답 내용 (요약): {preview}...")
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            print(f"[스킵] {symbol} - 데이터 형식 오류")
            return None
        rows = data["result"]["list"]
        if not rows:
            print(f"[스킵] {symbol} - 응답 list가 비어 있음")
            return None
        df = pd.DataFrame(rows)
        df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.astype({
            "timestamp": "int64", "open": "float", "high": "float",
            "low": "float", "close": "float", "volume": "float"
        })
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
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
        tickers = data["result"]["list"]
        return {item["symbol"]: float(item["lastPrice"]) for item in tickers if item["symbol"] in SYMBOLS}
    except Exception as e:
        print(f"[ERROR] 실시간 가격 조회 실패: {e}")
        return {}

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    df['trend_slope'] = df['ema10'].diff()
    df['percent_diff'] = (df['close"] - df['ma20']) / df['ma20']
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
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi)

    df["btc_dominance"] = get_btc_dominance()

    df = df.dropna()
    return df[[
        'close', 'volume', 'ma5', 'ma20', 'rsi', 'macd',
        'bollinger', 'volatility', 'trend_slope',
        'percent_diff', 'volume_delta',
        'obv', 'cci', 'stoch_rsi', 'btc_dominance'
    ]]
