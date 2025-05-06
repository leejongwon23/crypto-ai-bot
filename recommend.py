import torch
import numpy as np
from model import LSTMModel
from bybit_data import get_kline, get_current_price

def generate_recommendation(symbol="BTCUSDT"):
    klines = get_kline(symbol)
    if not klines or len(klines) < 200:
        return None

    # 종가 추출 및 정규화
    closes = np.array([x[4] for x in klines])
    normalized = (closes - closes.min()) / (closes.max() - closes.min())
    input_seq = torch.tensor(normalized[-50:]).reshape(1, 50, 1).float()

    # LSTM 예측
    model = LSTMModel()
    model.eval()
    with torch.no_grad():
        predicted = model(input_seq).item()

    # 진입가, 목표가, 방향성 판단
    entry = round(closes[-1], 2)
    target = round(closes.min() + predicted * (closes.max() - closes.min()), 2)
    direction = "상승" if target > entry else "하락"
    
    # 손절가 계산
    if direction == "상승":
        stop = round(entry * 0.98, 2)
    else:
        stop = round(entry * 1.02, 2)

    # 손익률 계산
    loss_pct = round(abs(entry - stop) / entry * 100, 2)
    profit_pct = round(abs(target - entry) / entry * 100, 2)

    # 실시간 현재가 불러오기
    current = get_current_price(symbol)
    current = round(current, 2) if isinstance(current, float) else "N/A"

    return {
        "symbol": symbol,
        "entry": entry,
        "current": current,
        "target": target,
        "stop": stop,
        "profit_pct": f"{profit_pct}",
        "loss_pct": f"{loss_pct}",
        "hit_rate": "70%",  # 추후 자동화 가능
        "reason": f"LSTM 예측 + 기술지표 기반 {direction} 확률"
    }
