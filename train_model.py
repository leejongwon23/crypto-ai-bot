import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from model import get_model
import os

# 예시용 CSV 파일 로드
df = pd.read_csv("sample_training_data.csv")  # 실제 CSV 경로로 변경

# 특성 엔지니어링
def compute_features(df):
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["bollinger"] = compute_bollinger(df["close"])
    df = df.dropna()
    return df[["close", "volume", "ma5", "ma20", "rsi", "macd", "bollinger"]].values

# 기술적 지표들 계산
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

# 시퀀스 생성
def make_sequences(data, window=30):
    sequences = []
    targets = []
    for i in range(len(data) - window - 1):
        seq = data[i:i+window]
        target = 1 if data[i+window][0] > data[i+window-1][0] else 0
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 전처리
features = compute_features(df)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
X, y = make_sequences(features_scaled, window=30)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 모델 불러오기 및 학습 준비
model = get_model(input_size=7)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 20
for epoch in range(epochs):
    model.train()
    outputs = torch.sigmoid(model(X_tensor))
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 저장
torch.save(model.state_dict(), "best_model.pt")
print("✅ 학습된 모델이 best_model.pt 로 저장되었습니다.")
