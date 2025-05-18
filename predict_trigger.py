# predict_trigger.py

import random
from data.utils import SYMBOLS, get_kline_by_strategy

# 전략별 최소 변동성 기준
MIN_VOLATILITY = {
    "단기": 0.005,
    "중기": 0.01,
    "장기": 0.015
}

def get_symbols_to_predict(strategy):
    selected = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 20:
                continue
            vol = df["close"].pct_change().rolling(20).std().iloc[-1]
            if vol is None or vol < MIN_VOLATILITY[strategy]:
                continue
            selected.append(symbol)
        except Exception as e:
            print(f"[오류] {symbol}-{strategy} 변동성 계산 실패: {e}")
    if not selected:
        # 아무 것도 선택되지 않으면 랜덤 5개라도 예측
        selected = random.sample(SYMBOLS, min(5, len(SYMBOLS)))
    return selected
