# train_model.py (왕1 보완 기능: 반복 학습 + 모델 저장)

import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from bybit_data import get_kline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# 코인 리스트 (왕1 기준 21개)
symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "LTCUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
    "ADAUSDT", "BNBUSDT", "ATOMUSDT", "NEARUSDT", "MATICUSDT",
    "APEUSDT", "SANDUSDT", "FTMUSDT", "EOSUSDT", "CHZUSDT", "ETCUSDT"
]

# 입력 특징 추출 함수 (recommend와 동일)
def compute_features(df):
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["boll"] = compute_bollinger(df["close"])
    df = df.dropna()
    return df[["close", "volume", "ma5", "ma20", "rsi", "macd", "boll"]].values

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    return ema_fast - ema_slow

def compute_bollinger(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - sma) / (2 * std)

def make_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window - 1):
        seq = data[i:i+window]
        target = 1 if data[i+window][0] > data[i+window-1][0] else 0
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def train_model():
    for symbol in symbols:
        for tf_name, interval in zip(["short", "mid", "long"], ["15", "60", "240"]):
            try:
                df = get_kline(symbol, interval)
                if df is None or len(df) < 100:
                    continue

                features = compute_features(df)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(features)
                X, y = make_sequences(scaled)

                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

                model = get_model(input_size=7)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # 학습 루프
                for epoch in range(10):
                    model.train()
                    output = torch.sigmoid(model(X_tensor))
                    loss = criterion(output, y_tensor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                os.makedirs("models", exist_ok=True)
                model_path = f"models/{symbol}_{tf_name}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"✅ 모델 저장됨: {model_path}")

            except Exception as e:
                print(f"❌ {symbol} [{tf_name}] 학습 실패: {e}")
