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

        # ✅ 데이터 유효성 검사
        if "result" not in data or "list" not in data["result"]:
            print(f"[에러] {symbol} 캔들 데이터 조회 실패: 'list'")
            return []

        rows = data["result"]["list"]
        if not rows or len(rows) < limit:
            print(f"[에러] {symbol} 캔들 데이터 부족 또는 비어있음")
            return []

        candles = []
        for row in rows:
            candles.append({
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5])
            })

        return candles

    except Exception as e:
        print(f"[에러] {symbol} 데이터 요청 실패: {e}")
        return []
