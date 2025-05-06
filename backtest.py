# backtest.py

import os
import torch
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from bybit_data import get_kline
from recommend import analyze_coin, get_model, extract_features, predict_with_model

# 백테스트 설정
symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
    "TRXUSDT", "LINKUSDT", "DOGEUSDT", "BCHUSDT", "STXUSDT", "SUIUSDT",
    "TONUSDT", "FILUSDT", "TRUMPUSDT", "HBARUSDT", "ARBUSDT", "APTUSDT",
    "UNISWAPUSDT", "BORAUSDT", "SANDUSDT"
]
target_datetime = datetime.datetime(2025, 5, 1, 9, 0)

# 모델 불러오기
model_path = "best_model.pt"

def backtest_symbol(symbol):
    candles = get_kline(symbol, interval=60, limit=240, end_time=target_datetime)
    if candles is None or len(candles) < 100:
        return None

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    features = extract_features(df)
    if len(features) < 30:
        return None

    X = features[-30:].values
    model = get_model(input_size=X.shape[1])

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("❌ 모델 파일이 없습니다.")
        return None

    prediction = predict_with_model(model, X)
    predicted_trend = 1 if prediction > 0.5 else 0

    # 실제 결과 비교 (분석 시간 이후 5개의 캔들 기준 상승/하락 판단)
    future_data = get_kline(symbol, interval=60, limit=5, end_time=target_datetime + datetime.timedelta(hours=5))
    if not future_data or len(future_data) < 5:
        return None

    entry_price = df["close"].values[-1]
    future_prices = [float(c[4]) for c in future_data]
    avg_future = sum(future_prices) / len(future_prices)
    real_trend = 1 if avg_future > entry_price else 0

    return predicted_trend, real_trend

# 전체 백테스트 실행
results = []
for symbol in symbols:
    try:
        result = backtest_symbol(symbol)
        if result:
            results.append(result)
            print(f"✅ {symbol} 예측: {result[0]} | 실제: {result[1]}")
    except Exception as e:
        print(f"⚠️ {symbol} 분석 실패: {e}")

# 정확도 계산
if results:
    y_pred, y_true = zip(*results)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n📊 총 테스트: {len(results)}개 | 예측 정확도: {round(acc * 100, 2)}%")
else:
    print("❌ 분석 결과 없음.")
