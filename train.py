import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features
from model.base_model import LSTMPricePredictor
from wrong_data_loader import load_wrong_prediction_data
import logger

DEVICE = torch.device("cpu")
WINDOW = 30
STOP_LOSS_PCT = 0.02

STRATEGY_GAIN_LEVELS = {
    "단기": [0.05, 0.07, 0.10],
    "중기": [0.10, 0.20, 0.30],
    "장기": [0.15, 0.30, 0.60]
}

def create_dataset(features, strategy, window=30):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        change = (future_close - current_close) / current_close

        levels = STRATEGY_GAIN_LEVELS[strategy]
        min_gain = levels[0]
        if abs(change) < min_gain or abs(change) > 1.0:
            continue
        label = 1 if change > 0 else 0
        if change <= -STOP_LOSS_PCT:
            continue

        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    return np.array(X), np.array(y)

def train_model(symbol, strategy, input_size=11, batch_size=32, epochs=10, lr=1e-3):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < WINDOW + 20:
        return

    df_feat = compute_features(df)
    if len(df_feat) < WINDOW + 1:
        return

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]

    X, y = create_dataset(feature_dicts, strategy, window=WINDOW)
    if len(X) == 0:
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    val_len = int(len(dataset) * 0.2)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = LSTMPricePredictor(input_size=input_size)
    model_path = f"models/{symbol}_{strategy}_lstm.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=WINDOW)
    if wrong_data:
        wrong_loader = DataLoader(wrong_data, batch_size=batch_size, shuffle=True)
        for xb, yb in wrong_loader:
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for epoch in range(epochs):
        for xb, yb in train_loader:
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)

def predict(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < WINDOW + 1:
        return None
    features = compute_features(df)
    if len(features) < WINDOW + 1:
        return None

    X = features.iloc[-WINDOW:].values
    X = np.expand_dims(X, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    model = LSTMPricePredictor(input_size=X.shape[2])
    model_path = f"models/{symbol}_{strategy}_lstm.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        signal, confidence = model(X_tensor)
        prob = signal.squeeze().item()
        confidence = confidence.squeeze().item()

    price = df["close"].iloc[-1]
    levels = STRATEGY_GAIN_LEVELS[strategy]
    direction = "롱" if prob > 0.5 else "숏"
    rate = levels[-1]
    if direction == "롱":
        target = price * (1 + rate)
        stop = price * (1 - STOP_LOSS_PCT)
    else:
        target = price * (1 - rate)
        stop = price * (1 + STOP_LOSS_PCT)

    rsi = features["rsi"].iloc[-1]
    macd = features["macd"].iloc[-1]
    boll = features["bollinger"].iloc[-1]
    reason = []
    if rsi < 30: reason.append("RSI 과매도")
    elif rsi > 70: reason.append("RSI 과매수")
    reason.append("MACD 상승 전환" if macd > 0 else "MACD 하락 전환")
    if boll > 1: reason.append("볼린저 상단 돌파")
    elif boll < -1: reason.append("볼린저 하단 이탈")

    return {
        "symbol": symbol,
        "strategy": strategy,
        "direction": direction,
        "price": price,
        "target": target,
        "stop": stop,
        "confidence": confidence,
        "rate": rate,
        "reason": ", ".join(reason)
    }

def get_price_now(symbol):
    from data.utils import get_realtime_prices
    prices = get_realtime_prices()
    return prices.get(symbol)

def main():
    logger.evaluate_predictions(get_price_now)
    for strategy in STRATEGY_GAIN_LEVELS:
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                if result:
                    logger.log_prediction(
                        symbol=result["symbol"],
                        strategy=result["strategy"],
                        direction=result["direction"],
                        entry_price=result["price"],
                        target_price=result["target"],
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=result["confidence"]
                    )
                    if result["confidence"] > 0.7:
                        msg = format_message(result)
                        send_message(msg)
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
