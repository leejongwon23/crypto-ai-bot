import torch
import numpy as np
from model import get_model
from bybit_data import get_kline
from sklearn.preprocessing import MinMaxScaler

# 30개 심볼
symbols = [
    "BTCUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "ETHUSDT",
    "XLMUSDT", "SUIUSDT", "ONDOUSDT", "LINKUSDT", "AVAXUSDT",
    "ETCUSDT", "UNIUSDT", "FILUSDT", "DOTUSDT", "LTCUSDT",
    "TRXUSDT", "FLOWUSDT", "STORJUSDT", "WAVESUSDT", "QTUMUSDT",
    "IOTAUSDT", "NEOUSDT", "DOGEUSDT", "SOLARUSDT", "TRUMPUSDT",
    "SHIBUSDT", "BCHUSDT", "SANDUSDT", "HBARUSDT", "GASUSDT"
]

window = 30

def compute_features(df):
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["boll"] = compute_bollinger(df["close"])
    df = df.dropna()
    return df[["close", "volume", "ma5", "ma20", "rsi", "macd", "boll"]]

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

def calculate_targets(entry_price, volatility=0.02):
    take_profit = entry_price * (1 + volatility * 1.5)
    stop_loss = entry_price * (1 - volatility)
    return take_profit, stop_loss

def explain_signal(rsi, macd, boll):
    reasons = []
    if rsi < 30:
        reasons.append("RSI 과매도")
    elif rsi > 70:
        reasons.append("RSI 과매수")

    if macd > 0:
        reasons.append("MACD 상승 전환")
    else:
        reasons.append("MACD 하락 전환")

    if boll > 1:
        reasons.append("볼린저 상단 돌파")
    elif boll < -1:
        reasons.append("볼린저 하단 이탈")

    return ", ".join(reasons)

def recommend_strategy():
    messages = []

    for symbol in symbols:
        try:
            df = get_kline(symbol, interval="60")
            if df is None or len(df) < 100:
                continue

            features = compute_features(df)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)
            X = []
            for i in range(len(scaled) - window):
                X.append(scaled[i:i+window])
            X = np.array(X)

            X_tensor = torch.tensor(X, dtype=torch.float32)
            model = get_model(input_size=7)
            model_path = f"models/{symbol}_mid_lstm.pt"
            model.load_state_dict(torch.load(model_path))
            model.eval()

            with torch.no_grad():
                preds = torch.sigmoid(model(X_tensor)).numpy().flatten()

            signal = "상승" if preds[-1] > 0.5 else "하락"
            entry_price = df["close"].iloc[-1]
            take_profit, stop_loss = calculate_targets(entry_price)

            rsi = features["rsi"].iloc[-1]
            macd = features["macd"].iloc[-1]
            boll = features["boll"].iloc[-1]
            reason = explain_signal(rsi, macd, boll)

            messages.append(
                f"📊 [{symbol}] 예측: {signal}\n"
                f"🟢 진입가: {entry_price:.2f}\n"
                f"🎯 목표가: {take_profit:.2f} / ❌ 손절가: {stop_loss:.2f}\n"
                f"🧠 진입사유: {reason}\n"
                f"📈 신뢰도: {preds[-1]*100:.2f}%"
            )

        except Exception as e:
            print(f"❌ {symbol} 분석 실패: {e}")

    return messages
