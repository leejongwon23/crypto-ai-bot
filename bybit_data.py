import requests

def get_kline(symbol="BTCUSDT", interval="60", limit=200):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    if data["retCode"] != 0:
        return None
    return [
        [float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])]
        for x in data["result"]["list"]
    ]
