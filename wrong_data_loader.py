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
            if len(row) < 4:
                continue
            _, sym, strat, direction, *_ = row
            if sym == symbol and strat == strategy:
                rows.append((row, direction))

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

        if any(len(row.values()) != len(feature_dicts[0].values()) for row in x_seq):
            continue

        # 🚩 예측 실패 row 목록에 포함되는지 확인
        for fail_row, direction in rows:
            try:
                entry_price = float(fail_row[4])
                timestamp = fail_row[0]
                close_price = df["close"].iloc[i + window - 1]
                ts_index = df.index[i + window - 1]
                if abs(close_price - entry_price) / entry_price < 0.001:
                    label = 1 if direction == "롱" else 0
                    X.append([list(r.values()) for r in x_seq])
                    y.append(label)
                    break
            except:
                continue

    if not X:
        return None

    seq_lens = [len(x) for x in X]
    mode_len = max(set(seq_lens), key=seq_lens.count)
    filtered = [(x, l) for x, l in zip(X, y) if len(x) == mode_len]
    if not filtered:
        return None

    X, y = zip(*filtered)
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)
