import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
import os

def create_dataset(features, window=20):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        if current_close == 0:
            continue
        change = (future_close - current_close) / current_close
        label = 1 if change > 0 else 0
        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    return np.array(X), np.array(y)

def find_best_window(symbol, strategy, window_list=[10, 20, 30, 40]):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < max(window_list) + 10:
        return 20  # 기본값

    df_feat = compute_features(df)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]

    best_score = -1
    best_window = window_list[0]

    for window in window_list:
        X, y = create_dataset(feature_dicts, window)
        if len(X) == 0:
            continue

        input_size = X.shape[2] if len(X.shape) == 3 else X.shape[1]
        model = get_model(model_type="lstm", input_size=input_size)
        model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        val_len = int(len(X_tensor) * 0.2)
        train_len = len(X_tensor) - val_len
        train_X = X_tensor[:train_len]
        train_y = y_tensor[:train_len]
        val_X = X_tensor[train_len:]
        val_y = y_tensor[train_len:]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        for _ in range(3):  # 빠르게 3 epoch만
            pred, _ = model(train_X)
            loss = criterion(pred.squeeze(), train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val, _ = model(val_X)
            pred_prob = pred_val.squeeze().numpy()
            pred_label = (pred_prob > 0.5).astype(int)
            acc = accuracy_score(val_y.numpy(), pred_label)
            conf = np.mean(np.abs(pred_prob - 0.5)) * 2  # confidence-like score

            score = acc * conf  # 정확도와 신뢰도 반영

            if score > best_score:
                best_score = score
                best_window = window

    save_path = f"/persistent/logs/best_window_{symbol}_{strategy}.txt"
    with open(save_path, "w") as f:
        f.write(str(best_window))

    print(f"[최적 WINDOW] {symbol}-{strategy} → {best_window} (score: {best_score:.4f})")
    return best_window
