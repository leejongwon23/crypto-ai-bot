import requests

# 1. 과거 캔들 데이터 가져오기 (1시간봉 기준, 200개)
def get_kline(symbol, interval=60, limit=200):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        data = response.json()
        return data["result"]["list"]
    except Exception as e:
        print(f"[에러] 캔들 조회 실패 ({symbol}): {e}")
        return None

# 2. 실시간 현재가 조회
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
