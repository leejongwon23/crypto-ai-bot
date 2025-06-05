import os
import csv
import torch
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from data.utils import get_kline_by_strategy, compute_features

def load_training_prediction_data(symbol, strategy, input_size, window=30, source_type="both"):
    files = []
    if source_type in ["correct", "both"]:
        files.append("/persistent/correct_predictions.csv")
    if source_type in ["wrong", "both"]:
        files.append("/persistent/wrong_predictions.csv")

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=14)

    seen, rows = set(), []
    for file in files:
        if not os.path.exists(file):
            continue
        with open(file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                timestamp, sym, strat, direction, entry_price, target_price, *_ = row
                if sym != symbol or strat != strategy:
                    continue
                try:
                    entry_price = float(entry_price)
                    dt = pd.to_datetime(timestamp, utc=True)

                    if direction not in ["롱", "숏"] or entry_price <= 0:
                        continue

                    key = (symbol, strategy, direction, entry_price)
                    if key in seen or dt < cutoff:
                        continue
                    seen.add(key)
                    rows.append({
                        "timestamp": dt,
                        "direction": direction,
                        "entry_price": entry_price
                    })
                except:
                    continue

    if not rows:
        return None

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < window + 1:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_feat = compute_features(symbol, df, strategy)
    if len(df_feat) < window + 1:
        return None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]

    samples = []
    for row in rows:
        try:
            fail_time = row["timestamp"]
            entry_price = row["entry_price"]
            direction = row["direction"]

            index = df[df["timestamp"] >= fail_time].index.min()
            if index is None or index < window:
                continue

            x_seq = feature_dicts[index - window:index]
            if any(len(r.values()) != len(feature_dicts[0].values()) for r in x_seq):
                continue

            future_df = df[df.index >= index]
            if future_df.empty:
                continue

            high = future_df["high"].max()
            low = future_df["low"].min()
            price = high if direction == "롱" else low
            gain = (price - entry_price) / entry_price if direction == "롱" else (entry_price - price) / entry_price

            if not np.isfinite(gain) or abs(gain) > 2:
                continue

            x = [list(r.values()) for r in x_seq]
            y = round(gain, 4)
            samples.append((np.array(x), y))
        except:
            continue

    if not samples:
        return None

    mode_len = max(set(len(s[0]) for s in samples), key=[s[0] for s in samples].count)
    filtered = [(x, y) for x, y in samples if len(x) == mode_len]
    return filtered if filtered else None

