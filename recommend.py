import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from model import get_model
from bybit_data import get_kline
import os


def extract_features(df):
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
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - sma) / (2 * std)

def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        return model(X_tensor).item()

def backtest_symbol(symbol, interval='60'):
    candles = get_kline(symbol, interval=interval)
    if not candles or len(candles) < 100:
        print(f"❌ 캔들 부족: {symbol}")
        return None

    df = pd.DataFrame(candles)
    if 'close' not in df or 'volume' not in df:
        return None

    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    features = extract_features(df)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    model = get_model(input_size=scaled.shape[1])
    if not os.path.exists("best_model.pt"):
        return None
    model.load_state_dict(torch.load("best_model.pt"))

    win = 0
    total = 0
    for i in range(30, len(scaled) - 1):
        X_input = scaled[i - 30:i]
        pred = predict(model, X_input)
        actual = df['close'].iloc[i + 1]
        prev = df['close'].iloc[i]

        direction = 1 if actual > prev else 0
        signal = 1 if pred > 0.5 else 0

        if direction == signal:
            win += 1
        total += 1

    accuracy = round((win / total) * 100, 2) if total > 0 else 0
    print(f"[백테스트] {symbol} 정확도: {accuracy}%")
    return {'symbol': symbol, 'accuracy': accuracy}


def run_backtest():
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
        "TRXUSDT", "LINKUSDT", "DOGEUSDT", "BCHUSDT", "SUIUSDT"
    ]

    results = []
    for sym in symbols:
        result = backtest_symbol(sym)
        if result:
            results.append(result)

    df_result = pd.DataFrame(results)
    df_result.to_csv("backtest_result.csv", index=False)
    print("✅ 백테스트 완료. 결과 저장됨: backtest_result.csv")


if __name__ == "__main__":
    run_backtest()
