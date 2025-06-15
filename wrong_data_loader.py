import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.utils import get_kline_by_strategy, compute_features

WRONG_CSV = "/persistent/wrong_predictions.csv"

def load_training_prediction_data(symbol, strategy, input_size, window):
    path = WRONG_CSV
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[불러오기 오류] {e}")
        return []

    df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "label"])  # ✅ label만 필수

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        return []

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty:
        return []

    if "timestamp" not in df_feat.columns:
        df_feat["timestamp"] = df_feat.get("datetime")

    df_feat = df_feat.dropna().reset_index(drop=True)
    sequences = []

    for _, row in df.iterrows():
        try:
            entry_time = row["timestamp"]
            label = int(float(row["label"]))  # ✅ predicted_class 대신 label 사용
            past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
            if len(past_window) < window:
                continue
            xb = past_window.drop(columns=["timestamp"]).to_numpy()
            if xb.shape != (window, input_size):
                continue
            sequences.append((xb, label))
        except Exception as e:
            print(f"[예외] 실패샘플 로딩 중 오류: {e}")
            continue

    return sequences
