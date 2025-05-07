# backtest.py (정확도 개선 로직 통합: 클래스 불균형 대응 + 민감도 측정)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

from model import get_model
from bybit_data import get_kline
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# 분석 대상 코인 (왕1 기준 21개)
symbols = [
    "BTCUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "ETHUSDT",
    "XLMUSDT", "SUIUSDT", "ONDOUSDT", "LINKUSDT", "AVAXUSDT",
    "ETCUSDT", "UNIUSDT", "FILUSDT", "DOTUSDT", "LTCUSDT",
    "TRXUSDT", "FLOWUSDT", "STORJUSDT", "WAVESUSDT", "QTUMUSDT",
    "IOTAUSDT", "NEOUSDT", "DOGEUSDT", "SOLARUSDT", "TRUMPUSDT",
    "SHIBUSDT", "BCHUSDT", "SANDUSDT", "HBARUSDT", "GASUSDT"
]
timeframe = "60"  # 1시간봉
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

def make_sequences(data):
    X, y = [], []
    for i in range(len(data) - window - 1):
        seq = data[i:i+window]
        target = 1 if data[i+window][0] > data[i+window-1][0] else 0
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def run_backtest():
    os.makedirs("logs", exist_ok=True)
    for symbol in symbols:
        try:
            df = get_kline(symbol, interval=timeframe)
            if df is None or len(df) < 100:
                print(f"⛔ {symbol} 데이터 부족")
                continue

            features = compute_features(df)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)
            X, y = make_sequences(scaled)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            model_path = f"models/{symbol}_mid_lstm.pt"
            model = get_model(7)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            preds = torch.sigmoid(model(X_tensor)).detach().numpy().flatten()
            pred_labels = (preds > 0.5).astype(int)
            accuracy = (pred_labels == y).mean()
            precision = precision_score(y, pred_labels)
            recall = recall_score(y, pred_labels)
            f1 = f1_score(y, pred_labels)

            print(f"✅ {symbol} 정확도: {accuracy*100:.2f}% / 정밀도: {precision:.2f} / 재현율: {recall:.2f} / F1: {f1:.2f}")

            # 그래프 저장
            plt.figure(figsize=(10, 4))
            plt.plot(preds, label="Predicted")
            plt.plot(y, label="Actual", alpha=0.5)
            plt.legend()
            plt.title(f"Backtest - {symbol}")
            plt.tight_layout()
            plt.savefig(f"logs/{symbol}_backtest.png")
            plt.close()

        except Exception as e:
            print(f"❌ {symbol} 처리 오류: {e}")

def schedule_backtest():
    scheduler = BlockingScheduler()
    scheduler.add_job(run_backtest, 'interval', hours=1)
    print("⏰ 자동 백테스트 시작됨 (1시간 주기)")
    scheduler.start()

if __name__ == "__main__":
    mode = input("실행 모드 선택 (1: 수동 실행, 2: 자동 백테스트): ")
    if mode == "1":
        run_backtest()
    elif mode == "2":
        schedule_backtest()
    else:
        print("❌ 잘못된 입력")
