import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from model import get_model
from bybit_data import get_training_data

# 기술적 지표 계산
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

# 학습 시퀀스 생성
def make_sequences(data, window=30):
    sequences = []
    targets = []
    for i in range(len(data) - window - 1):
        seq = data[i:i+window]
        target = 1 if data[i+window][0] > data[i+window-1][0] else 0
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 백테스트 실행 함수
def backtest_model(interval='60', model_path='best_model.pt'):
    df = get_training_data(interval=interval)
    if df is None or len(df) < 50:
        print("❌ 백테스트 데이터 부족")
        return

    features = compute_features(df)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    X, y = make_sequences(features_scaled, window=30)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = get_model(input_size=X.shape[2])
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print("❌ 모델 로딩 실패", e)
        return

    model.eval()
    with torch.no_grad():
        outputs = torch.sigmoid(model(X_tensor)).numpy().flatten()
    
    predictions = (outputs > 0.5).astype(int)
    accuracy = (predictions == y).mean() * 100
    print(f"✅ 백테스트 정확도 ({interval}): {accuracy:.2f}%")
