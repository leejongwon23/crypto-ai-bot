from model import CryptoPredictor
import torch
import numpy as np

def analyze_coin(symbol, candles, backtest=False):
    if len(candles) < 200:
        return None

    closes = np.array([c["close"] for c in candles])
    macds = np.array([c.get("macd", 0) for c in candles])
    boll_up = np.array([c.get("bollinger_upper", 0) for c in candles])
    volumes = np.array([c["volume"] for c in candles])

    X = np.stack([closes, macds, boll_up, volumes, closes * 0.9, closes * 1.1], axis=1)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    model = CryptoPredictor()
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).item()

    current_price = candles[-1]["close"]
    direction = "Long" if pred >= 0.5 else "Short"
    strategy_type = "단기상승" if direction == "Long" else "단기하락"
    target_price = current_price * (1.03 if direction == "Long" else 0.97)
    stop_loss = current_price * (0.97 if direction == "Long" else 1.03)
    expected_return = round(abs(target_price - current_price) / current_price * 100, 2)

    message = f"""
📌 코인: {symbol}
📈 진입가: {round(current_price, 3)} USDT
🎯 목표가: {round(target_price, 3)} USDT
🛑 손절가: {round(stop_loss, 3)} USDT
📊 전략: {strategy_type} / 예상 수익률: {expected_return}%
📅 분석 근거: macd, bollinger 기반 예측
"""
    return message.strip()
