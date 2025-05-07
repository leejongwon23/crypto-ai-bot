import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features

# 전략별 수익률 구간
STRATEGY_GAIN_LEVELS = {
    "단기": [0.05, 0.07, 0.10],
    "중기": [0.10, 0.20, 0.30],
    "장기": [0.15, 0.30, 0.60]
}
# 공통 손절 기준: -3%
MAX_LOSS = 0.03

def label_gain_class(current, future, strategy):
    levels = STRATEGY_GAIN_LEVELS[strategy]
    change = (future - current) / current
    if change <= -MAX_LOSS:
        return 0  # 손절
    for i, threshold in enumerate(levels, start=1):
        if change < threshold:
            return i
    return len(levels)  # 최고 구간 이상

def create_dataset(features, strategy, window=30):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']

        gain_class = label_gain_class(current_close, future_close, strategy)
        if gain_class == 0:
            continue  # 손절은 학습 제외

        X.append([list(row.values()) for row in x_seq])
        y.append(gain_class)
    return np.array(X), np.array(y)

def train_model(symbol, strategy, input_size=11, window=30, batch_size=32, epochs=10, lr=1e-3):
    gain_levels = STRATEGY_GAIN_LEVELS[strategy]
    print(f"📚 학습 시작: {symbol} / {strategy} / 수익률 구간: {gain_levels}, 손절 ≤ 3%")

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

    X, y = create_dataset(feature_dicts, strategy=strategy, window=window)
    if len(X) == 0:
        print(f"⚠️ 학습 불가: {symbol} / {strategy} 라벨 부족")
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    num_classes = len(gain_levels) + 1  # 클래스 수 (손절 제외하고 1~n)

    class MultiGainClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, num_classes=num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attn = nn.Linear(hidden_size, 1)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
            context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
            context = self.bn(context)
            context = self.dropout(context)
            return self.fc(context)

    model = MultiGainClassifier(input_size=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            output = model(xb)
            loss = criterion(output, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{symbol}-{strategy}] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{symbol}_{strategy}_gaincls.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ 저장 완료: {save_path}")

def main():
    while True:
        for strategy in STRATEGY_GAIN_LEVELS.keys():
            for symbol in SYMBOLS:
                try:
                    train_model(symbol, strategy)
                except Exception as e:
                    print(f"[ERROR] {symbol}-{strategy} 학습 오류: {e}")
        print("⏳ 1시간 대기 후 재학습 반복...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
