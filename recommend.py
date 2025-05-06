import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import get_model
import torch
import torch.nn as nn
import os

def extract_features(df):
    df = df.copy()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'] = compute_macd(df['close'])
    df['bollinger'] = compute_bollinger(df['close'])
    df = df.dropna()
    return df[['close', 'volume', 'ma5', 'ma20', 'rsi', 'macd', 'bollinger']]

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

def predict_with_model(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        prediction = model(X_tensor).item()
    return prediction

def recommend_strategy(df, model_path='best_model.pt'):
    df_feat = extract_features(df)
    if len(df_feat) < 30:
        return None

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_feat)

    X_input = X_scaled[-30:]
    model = get_model(input_size=X_input.shape[1])

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        return None

    prediction = predict_with_model(model, X_input)
    trend = "상승" if prediction > 0.5 else "하락"
    confidence = round(float(prediction) * 100, 2) if prediction > 0.5 else round((1 - float(prediction)) * 100, 2)
    return trend, confidence
