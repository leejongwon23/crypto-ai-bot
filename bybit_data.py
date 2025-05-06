import requests
import pandas as pd
import numpy as np

def get_kline(symbol, interval=60, limit=200, end_time=None):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    if end_time:
        url += f"&end={int(end_time.timestamp()) * 1000}"
    try:
        response = requests.get(url)
        data = response.json()
        raw = data['result']['list']
        candles = []
        for row in raw:
            candles.append({
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5])
            })
        return add_technical_indicators(candles)
    except:
        return None

def add_technical_indicators(candles):
    closes = np.array([c['close'] for c in candles])
    df = pd.Series(closes)
    ema12 = df.ewm(span=12).mean()
    ema26 = df.ewm(span=26).mean()
    candles[-1]['macd'] = float(ema12.iloc[-1] - ema26.iloc[-1])
    ma20 = df.rolling(window=20).mean()
    std20 = df.rolling(window=20).std()
    candles[-1]['bollinger_upper'] = float(ma20.iloc[-1] + 2 * std20.iloc[-1])
    return candles
