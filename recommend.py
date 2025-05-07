# recommend.py (signal_explainer + target_price_calc 통합 완료)

from bybit_data import get_kline
from model import get_model
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 📌 분석 대상 코인 21종
symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "LTCUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
    "ADAUSDT", "BNBUSDT", "ATOMUSDT", "NEARUSDT", "MATICUSDT",
    "APEUSDT", "SANDUSDT", "FTMUSDT", "EOSUSDT", "CHZUSDT", "ETCUSDT"
]

# ✅ 기술지표 계산 함수들

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

# 🎯 목표가 / 손절가 계산 (통합)
def calculate_targets(entry_price: float, volatility: float = 0.02):
    take_profit = entry_price * (1 + volatility * 1.5)
    stop_loss = entry_price * (1 - volatility)
    return round(take_profit, 2), round(stop_loss, 2)

# 💬 진입 사유 설명 (통합)
def explain_signals(row):
    explanations = []
    rsi = row.get("rsi", 50)
    if rsi < 30:
        explanations.append("📉 RSI 과매도 구간 접근")
    elif rsi > 70:
        explanations.append("📈 RSI 과매수 상태")
    macd = row.get("macd", 0)
    if macd > 0:
        explanations.append("🔺 MACD 상승 모멘텀")
    elif macd < 0:
        explanations.append("🔻 MACD 하락 모멘텀")
    boll = row.get("boll", 0)
    if boll > 1:
        explanations.append("⬆️ 밴드 상단 돌파")
    elif boll < -1:
        explanations.append("⬇️ 밴드 하단 이탈")
    return " / ".join(explanations) if explanations else "기술 지표 중립"

# 🔍 예측 수행

def predict(df, model):
    features = compute_features(df)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    if len(scaled) < 31:
        return None, None
    window = 30
    seq = scaled[-window:]
    x = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)
    pred = model(x)
    prob = torch.sigmoid(pred).item()
    latest_raw = features[-1:][0]
    latest_row = dict(zip(["close", "volume", "ma5", "ma20", "rsi", "macd", "boll"], latest_raw))
    reason = explain_signals(latest_row)
    return prob, reason

# 📊 전략 실행
def recommend_strategy():
    result_msgs = []

    for symbol in symbols:
        try:
            df_short = get_kline(symbol, interval="15")
            df_mid = get_kline(symbol, interval="60")
            df_long = get_kline(symbol, interval="240")
            if df_short is None or df_mid is None or df_long is None:
                continue

            last_price = round(df_short["close"].iloc[-1], 2)
            tp, sl = calculate_targets(last_price)

            result_set = [
                ("단기", df_short, f"models/{symbol}_short.pt"),
                ("중기", df_mid, f"models/{symbol}_mid.pt"),
                ("장기", df_long, f"models/{symbol}_long.pt")
            ]

            for label, df, model_path in result_set:
                model = get_model(7)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                prob, reason = predict(df, model)
                if prob is None:
                    continue
                trend = "📈 상승" if prob > 0.5 else "📉 하락"
                confidence = round(prob * 100, 2)
                msg = (
                    f"📌 {symbol} ({label})\n"
                    f"진입가: {last_price} USDT\n"
                    f"목표가: {tp} / 손절가: {sl}\n"
                    f"신뢰도: {confidence}%\n"
                    f"예측: {trend}\n"
                    f"사유: {reason}"
                )
                result_msgs.append(msg)

        except Exception as e:
            print(f"❌ {symbol} 처리 오류: {e}")
            continue

    return result_msgs


