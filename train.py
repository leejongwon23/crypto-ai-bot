import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features
from model.base_model import LSTMPricePredictor

# 전략별 최소 수익률 설정 (단기: 5%, 중기: 10%, 장기: 15%)
STRATEGY_TARGETS = {
    "단기": {"min_gain": 0.05},
    "중기": {"min_gain": 0.10},
    "장기": {"min_gain": 0.15}
}

def create_dataset(features, window=30, min_gain=0.05):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        future_close = features[i+window]['close']
        current_close = features[i+window-1]['close']
        label = 1 if future_close >= current_close * (1 + min_gain) else 0
        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    return np.array(X), np.array(y)

def train_model(symbol, strategy, input_size=11, window=30, batch_size=32, epochs=10, lr=1e-3):
    min_gain = STRATEGY_TARGETS.get(strategy, {}).get("min_gain", 0.05)
    print(f"📚 학습 시작: {symbol} / {strategy} / 최소 목표 수익률: {min_gain*100:.1f}%")

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < window + 20:
        print(f"❌ {symbol} / {strategy} 데이터 부족")
        return

    df_feat = compute_features(df)
    if len(df_feat) < window + 1:
        print(f"❌ {symbol} / {strategy} 피처 부족")
        return

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]

    X, y = create_dataset(feature_dicts, window=window, min_gain=min_gain)
    if len(X) == 0:
        print(f"⚠️ 학습 불가: {symbol} / {strategy} 라벨 부족")
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = LSTMPricePredictor(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            signal_pred, _ = model(xb)
            loss = criterion(signal_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{symbol}-{strategy}] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{symbol}_{strategy}_lstm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ 저장 완료: {save_path}")

def main():
    while True:
        for strategy in STRATEGY_CONFIG.keys():
            for symbol in SYMBOLS:
                try:
                    train_model(symbol, strategy)
                except Exception as e:
                    print(f"[ERROR] {symbol}-{strategy} 학습 중 오류: {e}")
        print("🕐 모든 전략 학습 완료. 1시간 대기 후 재시작...")
        time.sleep(3600)  # 1시간마다 반복

if __name__ == "__main__":
    main()
