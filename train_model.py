# train_model.py (통합: accuracy_logger + logger + train_model_4H + train_model_gru)

import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model, get_gru_model  # GRU 모델 지원
from bybit_data import get_kline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import csv
from datetime import datetime

# 코인 리스트 (왕1 기준 21개)
symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "LTCUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
    "ADAUSDT", "BNBUSDT", "ATOMUSDT", "NEARUSDT", "MATICUSDT",
    "APEUSDT", "SANDUSDT", "FTMUSDT", "EOSUSDT", "CHZUSDT", "ETCUSDT"
]

# 기술지표 계산 함수
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

# 정확도 로깅

def log_prediction(symbol, timeframe, true_label, predicted_prob):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "prediction_accuracy.csv")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    predicted_label = 1 if predicted_prob >= 0.5 else 0
    correct = 1 if predicted_label == true_label else 0

    row = [now, symbol, timeframe, true_label, predicted_label, predicted_prob, correct]
    header = ["timestamp", "symbol", "timeframe", "true_label", "predicted_label", "predicted_prob", "correct"]
    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"✅ {symbol} [{timeframe}] 로그 기록 완료: 정확도={correct}, 확률={predicted_prob:.2f}")

# 모델 학습

def train_model(use_gru=False, only_4h=False):
    for symbol in symbols:
        timeframes = [("short", "15"), ("mid", "60"), ("long", "240")]
        if only_4h:
            timeframes = [("long", "240")]

        for tf_name, interval in timeframes:
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

                model = get_gru_model(7) if use_gru else get_model(7)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(10):
                    model.train()
                    output = torch.sigmoid(model(X_tensor))
                    loss = criterion(output, y_tensor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                os.makedirs("models", exist_ok=True)
                model_name = "gru" if use_gru else "lstm"
                model_path = f"models/{symbol}_{tf_name}_{model_name}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"✅ 모델 저장됨: {model_path}")

                # 정확도 로깅
                with torch.no_grad():
                    preds = torch.sigmoid(model(X_tensor)).numpy().flatten()
                    for i in range(len(y)):
                        log_prediction(symbol, tf_name, int(y[i]), float(preds[i]))

            except Exception as e:
                print(f"❌ {symbol} [{tf_name}] 학습 실패: {e}")
