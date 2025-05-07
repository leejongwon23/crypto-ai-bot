from bybit_data import get_kline
from model import get_model
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "LTCUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
    "ADAUSDT", "BNBUSDT", "ATOMUSDT", "NEARUSDT", "MATICUSDT",
    "APEUSDT", "SANDUSDT", "FTMUSDT", "EOSUSDT", "CHZUSDT", "ETCUSDT"
]

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

def get_targets(entry):
    return None, None  # 고정 수익률 제한 제거됨

def predict(df, model):
    features = compute_features(df)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    if len(scaled) < 31:
        return None
    seq = scaled[-30:]
    x = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)
    pred = model(x)
    prob = torch.sigmoid(pred).item()
    return prob

def recommend_strategy():
    result_msgs = []
    for symbol in symbols:
        try:
            df_short = get_kline(symbol, interval="15")
            df_mid = get_kline(symbol, interval="60")
            df_long = get_kline(symbol, interval="240")
            if df_short is None or df_mid is None or df_long is None:
                continue

            model_s = get_model(7)
            model_s.load_state_dict(torch.load(f"models/{symbol}_short.pt"))
            model_s.eval()
            prob_s = predict(df_short, model_s)

            model_m = get_model(7)
            model_m.load_state_dict(torch.load(f"models/{symbol}_mid.pt"))
            model_m.eval()
            prob_m = predict(df_mid, model_m)

            model_l = get_model(7)
            model_l.load_state_dict(torch.load(f"models/{symbol}_long.pt"))
            model_l.eval()
            prob_l = predict(df_long, model_l)

            last_price = round(df_short["close"].iloc[-1], 2)

            for label, prob in zip(["단기", "중기", "장기"], [prob_s, prob_m, prob_l]):
                if prob is None:
                    continue
                trend = "📈 상승" if prob > 0.5 else "📉 하락"
                confidence = round(prob * 100, 2)
                msg = (
                    f"📌 {symbol} ({label})\n"
                    f"진입가: {last_price} USDT\n"
                    f"목표가: 전략별 수익률 설정 필요\n"
                    f"손절가: 전략별 리스크 기준 설정 필요\n"
                    f"신뢰도: {confidence}%\n"
                    f"예측: {trend}"
                )
                result_msgs.append(msg)
        except Exception as e:
            print(f"❌ {symbol} 처리 오류: {e}")
            continue
    return result_msgs

