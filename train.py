import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features
from model.base_model import LSTMPricePredictor
from wrong_data_loader import load_wrong_prediction_data  # 오답 학습용 보조 모듈

WINDOW = 30
GAIN_RANGES = {
    "단기": (0.05, 0.15),
    "중기": (0.10, 0.30),
    "장기": (0.20, 1.00)
}
MAX_LOSS = 0.02  # 손절가 -2%

def label_gain_class(current, future, strategy):
    change = (future - current) / current
    min_gain, max_gain = GAIN_RANGES[strategy]
    if abs(change) < min_gain or abs(change) > max_gain:
        return 0
    return 1 if change > 0 else 0

def create_dataset(features, strategy, window=30):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        change = (future_close - current_close) / current_close

        # 전략별 수익률 범위 반영
        label = label_gain_class(current_close, future_close, strategy)
        if label is None:
            continue

        # 손절 -2% 이상인 경우 제거
        if change <= -MAX_LOSS:
            continue

        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    return np.array(X), np.array(y)

def train_model(symbol, strategy, input_size=11, batch_size=32, epochs=10, lr=1e-3):
    print(f"📚 학습 시작: {symbol} / {strategy}")

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < WINDOW + 20:
        print(f"❌ {symbol} / {strategy} 데이터 부족")
        return

    df_feat = compute_features(df)
    if len(df_feat) < WINDOW + 1:
        print(f"❌ {symbol} / {strategy} 피처 부족")
        return

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]

    X, y = create_dataset(feature_dicts, strategy, window=WINDOW)
    if len(X) == 0:
        print(f"⚠️ 라벨 부족: {symbol} / {strategy}")
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    val_len = int(len(dataset) * 0.2)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = LSTMPricePredictor(input_size=input_size)
    model_path = f"models/{symbol}_{strategy}_lstm.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"📦 이전 모델 로드: {model_path}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    # 오답 학습 우선 처리
    wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=WINDOW)
    if wrong_data:
        print(f"⚠️ 오답 우선 학습 실행 중: {symbol} / {strategy}")
        wrong_loader = DataLoader(wrong_data, batch_size=batch_size, shuffle=True)
        for xb, yb in wrong_loader:
            signal_pred, _ = model(xb)
            loss = criterion(signal_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 일반 데이터 학습
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            signal_pred, _ = model(xb)
            loss = criterion(signal_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{symbol}-{strategy}] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ 모델 저장 완료: {model_path}")

def main():
    while True:
        for strategy in STRATEGY_CONFIG:
            for symbol in SYMBOLS:
                try:
                    train_model(symbol, strategy)
                except Exception as e:
                    print(f"[ERROR] {symbol}-{strategy} 학습 오류: {e}")
        print("⏳ 1시간 대기 후 재학습 반복...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
