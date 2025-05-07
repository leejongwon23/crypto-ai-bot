import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from model import get_model
from bybit_data import get_training_data
import os

# ✅ 기술적 지표 계산
def compute_features(df):
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["bollinger"] = compute_bollinger(df["close"])
    df = df.dropna()
    return df[["close", "volume", "ma5", "ma20", "rsi", "macd", "bollinger"]].values

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
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - sma) / (2 * std)

# ✅ 시퀀스 데이터 생성
def make_sequences(data, window=30):
    sequences = []
    targets = []
    for i in range(len(data) - window - 1):
        seq = data[i:i+window]
        target = 1 if data[i+window][0] > data[i+window-1][0] else 0
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# ✅ 주기별 학습 함수 (1시간봉, 4시간봉, 일봉 확장 가능)
def train_model_with_interval(interval='60', model_path='best_model.pt'):
    df = get_training_data(interval=interval)
    if df is None or len(df) < 50:
        print("❌ 학습 데이터 부족")
        return

    features = compute_features(df)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    X, y = make_sequences(features_scaled, window=30)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = get_model(input_size=X.shape[2])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        outputs = torch.sigmoid(model(X_tensor))
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[Interval: {interval}] Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ 모델 저장됨: {model_path}")
