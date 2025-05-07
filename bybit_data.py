import requests

def get_kline(symbol, interval="60", limit=200):
    try:
        url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url)
        data = response.json()

        if data["retCode"] != 0:
            print(f"[에러] {symbol} 캔들 데이터 조회 실패: {data['retMsg']}")
            return []

        rows = data["result"]["list"]
        rows.reverse()  # 시간 순으로 정렬 (가장 오래된 것부터)

        return [
            {
                "timestamp": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5])
            }
            for r in rows
        ]

    except Exception as e:
        print(f"[예외 발생] {symbol} 처리 중 오류: {e}")
        return []
