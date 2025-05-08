import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features

STRATEGY_GAIN_LEVELS = {
    "단기": [0.05, 0.07, 0.10],
    "중기": [0.10, 0.20, 0.30],
    "장기": [0.15, 0.30, 0.60]
}
MIN_MAX_GAIN = {
    "단기": (0.05, 0.15),
    "중기": (0.10, 0.30),
    "장기": (0.20, 1.00)
}
MAX_LOSS = 0.02
WINDOW = 30
DAYS_LOOKBACK = 7
VALID_RATIO = 0.2

def label_gain_class(current, future, strategy):
    levels = STRATEGY_GAIN_LEVELS[strategy]
    min_gain, max_gain = MIN_MAX_GAIN[strategy]
    change = (future - current) / current
    if abs(change) < min_gain or abs(change) > max_gain:
        return 0
    for i, threshold in reversed(list(enumerate(levels, start=1))):
        if change <= -threshold:
            return len(levels) + i
    for i, threshold in enumerate(levels, start=1):
        if change >= threshold:
            return i
    return 0

def create_dataset(features, strategy, window=30):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        label = label_gain_class(current_close, future_close, strategy)
        if label == 0:
            continue
        X.append([list(row.values()) for row in x_seq])
        y.append(label - 1)
    return np.array(X), np.array(y)

def collect_extended_data(symbol, strategy):
    total_df = []
    for _ in range(DAYS_LOOKBACK):
        df = get_kline_by_strategy(symbol, strategy)
        if df is not None:
            total_df.append(df)
        time.sleep(0.5)
    if not total_df:
        return None
    df_all = total_df[0]
    for d in total_df[1:]:
        df_all = df_all.append(d)
    df_all = df_all.sort_values("datetime").reset_index(drop=True)
    return df_all

def train_model(symbol, strategy, input_size=11, window=30, batch_size=32, epochs=10, lr=1e-3):
    gain_levels = STRATEGY_GAIN_LEVELS[strategy]
    num_classes = len(gain_levels) * 2
    print(f"📚 학습 시작: {symbol} / {strategy} / 클래스 수: {num_classes}")

    df = collect_extended_data(symbol, strategy)
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
        print(f"⚠️ 라벨 부족: {symbol} / {strategy}")
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_len = int(len(dataset) * VALID_RATIO)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    class DualGainClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, num_classes=num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attn = nn.Linear(hidden_size, 1)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
            context = torch.sum(lstm_out * w.unsqueeze(-1), dim=1)
            context = self.bn(context)
            context = self.drop(context)
            return self.fc(context)

    model = DualGainClassifier(input_size=input_size)
    save_path = f"models/{symbol}_{strategy}_dual.pt"
    if os.path.exists(save_path):
        print(f"📦 이전 모델 로드: {save_path}")
        model.load_state_dict(torch.load(save_path, map_location='cpu'))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            output = model(xb)
            loss = criterion(output, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{symbol}-{strategy}] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Validation 평가
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(yb.numpy())
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"✅ 검증 정확도: {acc:.4f} / 정밀도: {prec:.4f} / 재현율: {rec:.4f}")

    # 모델 저장 (기본 + 백업)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    timestamp = time.strftime("%Y%m%d_%H%M")
    backup_path = f"models/{symbol}_{strategy}_dual_{timestamp}.pt"
    torch.save(model.state_dict(), backup_path)
    print(f"✅ 모델 저장: {save_path} / 백업: {backup_path}")

def main():
    while True:
        for strategy in STRATEGY_GAIN_LEVELS:
            for symbol in SYMBOLS:
                try:
                    train_model(symbol, strategy)
                except Exception as e:
                    print(f"[ERROR] {symbol}-{strategy} 학습 오류: {e}")
        print("⏳ 1시간 대기 후 재학습 반복...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
