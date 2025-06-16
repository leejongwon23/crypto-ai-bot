import os
import pandas as pd
import numpy as np
from data.utils import get_kline_by_strategy, compute_features

WRONG_CSV = "/persistent/wrong_predictions.csv"

def load_training_prediction_data(symbol, strategy, input_size, window):
    if not os.path.exists(WRONG_CSV):
        return []

    try:
        df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig")
        df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "label"])
    except Exception as e:
        print(f"[불러오기 오류] {symbol}-{strategy} → {e}")
        return []

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        print(f"[스킵] {symbol}-{strategy} → 시세데이터 없음")
        return []

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty:
        print(f"[스킵] {symbol}-{strategy} → 피처 생성 실패")
        return []

    if df_feat.isnull().any().any():
        print(f"[스킵] {symbol}-{strategy} → NaN 포함 피처 제거")
        return []

    if "timestamp" not in df_feat.columns:
        df_feat["timestamp"] = df_feat.get("datetime")
    df_feat = df_feat.dropna().reset_index(drop=True)

    sequences = []
    for _, row in df.iterrows():
        try:
            entry_time = row["timestamp"]
            label = int(float(row["label"]))

            past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
            if len(past_window) < window:
                continue

            xb = past_window.drop(columns=["timestamp"]).to_numpy(dtype=np.float32)
            if xb.shape != (window, input_size):
                continue

            sequences.append((xb, label))

        except Exception as e:
            print(f"[예외] {symbol}-{strategy} 실패샘플 처리 오류 → {e}")
            continue

    return sequences
