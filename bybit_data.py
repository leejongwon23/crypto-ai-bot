import requests
import numpy as np

def get_kline(symbol):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=60&limit=200"
    try:
        res = requests.get(url)
        data = res.json()
        items = data["result"]["list"]
        result = []

        closes = [float(i[4]) for i in items]
        volumes = [float(i[5]) for i in items]

        # MA20
        ma20 = np.convolve(closes, np.ones(20) / 20, mode='valid')
        ma20 = [None]*(len(closes)-len(ma20)) + list(ma20)

        # RSI
        rsi = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            up = max(diff, 0)
            down = max(-diff, 0)
            rsi.append((up, down))

        rsi_values = []
        for i in range(14, len(rsi)):
            avg_gain = np.mean([r[0] for r in rsi[i-14:i]])
            avg_loss = np.mean([r[1] for r in rsi[i-14:i]])
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi_values.append(100 - (100 / (1 + rs)))
        rsi = [None]*15 + rsi_values

        for i in range(len(closes)):
            if None in (ma20[i], rsi[i]):
                continue
            result.append([closes[i], volumes[i], ma20[i], rsi[i]])
        return result
    except Exception as e:
        print(f"[에러] 캔들 데이터 조회 실패 ({symbol}): {e}")
        return []

def get_current_price(symbol):
    url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
    try:
        res = requests.get(url)
        data = res.json()
        price = float(data["result"]["list"][0]["lastPrice"])
        return price
    except Exception as e:
        print(f"[에러] 현재가 조회 실패 ({symbol}): {e}")
        return "N/A"
