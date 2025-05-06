# bybit_data.py
import requests
import pandas as pd
import numpy as np

def get_kline(symbol, interval=60, limit=200, end_time=None):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    if end_time:
        url += f"&end={int(end_time.timestamp())}"

    try:
        response = requests.get(url)
        data = response.json()

        # ✅ 데이터 유효성 검사
        if "result" not in data or "list" not in data["result"]:
            print(f"[에러] {symbol} 캔들 데이터 조회 실패: 'result' 또는 'list' 누락")
            return []

        rows = data["result"]["list"]
        if not rows or len(rows) < limit:
            print(f"[에러] {symbol} 캔들 데이터 부족: {len(rows)}개")
            return []

        # ✅ DataFrame 생성
        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df

    except Exception as e:
        print(f"[예외 발생] {symbol} 처리 중 오류: {e}")
        return []
