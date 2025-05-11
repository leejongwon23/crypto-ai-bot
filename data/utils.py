import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.bybit.com"

SYMBOLS = [
    "BTCUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "ETHUSDT",
    "XLMUSDT", "SUIUSDT", "ONDOUSDT", "LINKUSDT", "AVAXUSDT",
    "ETCUSDT", "UNIUSDT", "FILUSDT", "DOTUSDT", "LTCUSDT",
    "TRXUSDT", "FLOWUSDT", "STORJUSDT", "WAVESUSDT", "QTUMUSDT",
    "IOTAUSDT", "NEOUSDT", "DOGEUSDT", "SOLARUSDT", "TRUMPUSDT",
    "SHIBUSDT", "BCHUSDT", "SANDUSDT", "HBARUSDT", "GASUSDT"
]

STRATEGY_CONFIG = {
    "단기": {"interval": "4h", "limit": 300},
    "중기": {"interval": "1d", "limit": 365},
    "장기": {"interval": "1w", "limit": 300}
}

def get_kline(symbol: str, interval: str = "60", limit: int = 200):
    url = f"{BASE_URL}/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=10)
        print(f"[DEBUG] {symbol} 요청 URL: {res.url}")
        print(f"[DEBUG] {symbol} 응답 내용: {res.text}")  # ✅ 응답 확인
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

def filter_by_volume(df: pd.DataFrame, min_volume: float = 1000000):
    recent_volume = df["volume"].iloc[-1]
    return recent_volume >= min_volume

def get_long_short_ratio(symbol: str):
    url = f"{BASE_URL}/v5/market/account-ratio"
    params = {"category": "linear", "symbol": symbol, "period": "1h", "limit": 1}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return None
        ratio_data = data["result"]["list"][0]
        return {
            "long": float(ratio_data["buyRatio"]),
            "short": float(ratio_data["sellRatio"])
        }
    except Exception as e:
        print(f"[ERROR] {symbol} 롱숏 비율 조회 실패: {e}")
        return None

def get_trade_strength(symbol: str):
    url = f"{BASE_URL}/v5/market/public-trading-history"
    params = {"category": "linear", "symbol": symbol, "limit": 200}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return None
        trades = data["result"]["list"]
        buy_count = sum(1 for t in trades if t["side"] == "Buy")
        sell_count = sum(1 for t in trades if t["side"] == "Sell")
        total = buy_count + sell_count
        if total == 0:
            return None
        return {
            "buy_ratio": buy_count / total,
            "sell_ratio": sell_count / total
        }
    except Exception as e:
        print(f"[ERROR] {symbol} 체결강도 조회 실패: {e}")
        return None

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
    df['percent_diff'] = (df['close'] - df['ma20']) / df['ma20']
    df['volume_delta'] = df['volume'].diff()
    df = df.dropna()
    return df[[
        'close', 'volume', 'ma5', 'ma20', 'rsi', 'macd',
        'bollinger', 'volatility', 'trend_slope',
        'percent_diff', 'volume_delta'
    ]]
