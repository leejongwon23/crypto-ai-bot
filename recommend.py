import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import get_ensemble_models
import torch
import os
from bybit_data import get_kline

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

def predict_with_ensemble(models, X):
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            pred = model(X_tensor).item()
            preds.append(pred)
    return sum(preds) / len(preds)

def recommend_strategy(df, model_paths=['best_model_lstm.pt', 'best_model_gru.pt']):
    df_feat = extract_features(df)
    if len(df_feat) < 30:
        return None

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_feat)
    X_input = X_scaled[-30:]

    models = get_ensemble_models(input_size=X_input.shape[1])
    for model, path in zip(models, model_paths):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
        else:
            return None

    prediction = predict_with_ensemble(models, X_input)
    trend = "📈 상승" if prediction > 0.5 else "📉 하락"
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
    return trend, confidence

def recommend_all():
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
        "TRXUSDT", "LINKUSDT", "DOGEUSDT", "BCHUSDT", "STXUSDT", "SUIUSDT",
        "TONUSDT", "FILUSDT", "TRUMPUSDT", "HBARUSDT", "ARBUSDT", "APTUSDT",
        "UNIUSMARGUSDT", "BORAUSDT", "SANDUSDT"
    ]

    messages = []
    for symbol in symbols:
        try:
            candles = get_kline(symbol)
            if not candles or len(candles) < 100:
                continue

            df = pd.DataFrame(candles)
            if 'volume' not in df.columns or 'close' not in df.columns:
                continue

            df["volume"] = df["volume"].astype(float)
            df["close"] = df["close"].astype(float)

            result = recommend_strategy(df)
            if result:
                trend, confidence = result
                entry_price = round(float(df["close"].iloc[-1]), 4)
                if trend == "📈 상승":
                    target_price = round(entry_price * 1.03, 4)
                    stop_price = round(entry_price * 0.98, 4)
                else:
                    target_price = round(entry_price * 0.97, 4)
                    stop_price = round(entry_price * 1.02, 4)

                msg = (
                    f"<b>{symbol}</b>\n"
                    f"예측: {trend} / 신뢰도: {confidence}%\n"
                    f"📍 진입가: {entry_price}\n🎯 목표가: {target_price}\n⛔ 손절가: {stop_price}"
                )
                messages.append(msg)
        except Exception as e:
            print(f"⚠️ {symbol} 오류: {e}")
            continue

    return messages
