# backtest.py

from bybit_data import get_kline
from recommend import analyze_coin
import datetime

# 백테스트 대상 심볼 리스트
symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
    "TRXUSDT", "LINKUSDT", "DOGEUSDT", "BCHUSDT", "STXUSDT", "SUIUSDT",
    "TONUSDT", "FILUSDT", "TRUMPUSDT", "HBARUSDT", "ARBUSDT", "APTUSDT",
    "UNISWAPUSDT", "BORAUSDT", "SANDUSDT"
]

# 분석할 백테스트 날짜 (예: 2025-05-01 09:00)
target_datetime = datetime.datetime(2025, 5, 1, 9, 0)

# 유사한 구조로 분석
for symbol in symbols:
    try:
        candles = get_kline(symbol, interval=60, limit=200, end_time=target_datetime)
        if candles is None or len(candles) < 200:
            continue
        msg = analyze_coin(symbol, candles, backtest=True)
        print(f"[{symbol}] 분석결과:\n{msg}")
    except Exception as e:
        print(f"{symbol} 분석 중 오류 발생: {e}")
