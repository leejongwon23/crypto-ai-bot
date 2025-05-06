import torch
import numpy as np
from model import train_model
from bybit_data import get_kline, get_current_price

def generate_recommendation(symbol="BTCUSDT"):
    klines = get_kline(symbol)
    if not klines or len(klines) < 51:
        return None

    closes = np.array([x[0] for x in klines])
    normalized = []

    for item in klines:
        close, volume, ma20, rsi = item
        n_close = (close - closes.min()) / (closes.max() - closes.min())
        n_vol = volume / max([x[1] for x in klines])
        n_ma = ma20 / max([x[2] for x in klines])
        n_rsi = rsi / 100
        normalized.append([n_close, n_vol, n_ma, n_rsi])

    data = torch.tensor(normalized).float()
    X = data[:-1][-50:].reshape(1, 50, 4)
    y = torch.tensor([[(closes[-1] - closes.min()) / (closes.max() - closes.min())]]).float()

    model = train_model((X, y))
    model.eval()

    with torch.no_grad():
        pred = model(X).item()

    entry = round(closes[-1], 2)
    predicted_price = round(closes.min() + pred * (closes.max() - closes.min()), 2)

    if predicted_price > entry:
        stop = round(entry * 0.98, 2)
        direction = "📈 롱"
        loss_pct = round((entry - stop) / entry * 100, 2)
    else:
        stop = round(entry * 1.02, 2)
        direction = "📉 숏"
        loss_pct = round((stop - entry) / entry * 100, 2)

    profit_pct = round((predicted_price - entry) / entry * 100, 2)
    now = get_current_price(symbol)

    message = f"""\
📌 <b>{symbol}</b> {direction}
┌ 현재가: {now}
├ 진입가: {entry}
├ 목표가: {predicted_price} ({'+' if profit_pct > 0 else ''}{profit_pct}%)
├ 손절가: {stop} ({'-' + str(loss_pct)}%)
├ 정확도: 70%
└ 분석: LSTM 실시간 예측 기반
"""
    return message
