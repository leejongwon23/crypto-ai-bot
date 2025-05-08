import os
import csv
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from data.utils import get_kline_by_strategy, compute_features

def load_wrong_prediction_data(symbol, strategy, input_size, window=30):
    file = "wrong_predictions.csv"
    if not os.path.exists(file):
        return None

    rows = []
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            _, sym, strat, _, _, _, _, _ = row
            if sym == symbol and strat == strategy:
                rows.append(row)

    if not rows:
        return None

    df = get_kline_by_strategy(symbol, strategy)
    if df is None:
        return None

    df_feat = compute_features(df)
    if len(df_feat) < window + 1:
        return None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]

    X, y = [], []
    for i in range(len(feature_dicts) - window - 1):
        x_seq = feature_dicts[i:i+window]
        current_close = feature_dicts[i+window-1]['close']
        future_close = feature_dicts[i+window]['close']
        change = (future_close - current_close) / current_close
        label = 1 if change > 0 else 0
        X.append([list(r.values()) for r in x_seq])
        y.append(label)

    if not X:
        return None

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)
