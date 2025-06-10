import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.utils import get_kline_by_strategy, compute_features

WRONG_CSV = "/persistent/wrong_predictions.csv"

def load_training_prediction_data(symbol, strategy, input_size, window, source_type="wrong", min_samples=3):
    if not os.path.exists(WRONG_CSV):
        print(f"[불러오기 실패] wrong_predictions.csv 없음")
        return []

    try:
        df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig")
    except Exception as e:
        print(f"[불러오기 오류] {e}")
        return []

    df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "predicted_class"])

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        print(f"[가격 데이터 없음] {symbol}-{strategy}")
        return []

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty:
        print(f"[피처 생성 실패] {symbol}-{strategy}")
        return []

    if "timestamp" not in df_feat.columns:
        df_feat["timestamp"] = df_feat.get("datetime")

    df_feat = df_feat.dropna().reset_index(drop=True)
    sequences = []

    for _, row in df.iterrows():
        try:
            entry_time = row["timestamp"]
            predicted_class = int(row["predicted_class"])

            past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
            if len(past_window) < window:
                continue

            features_only = past_window.drop(columns=["timestamp"])
            xb = features_only.to_numpy()
            if xb.shape != (window, input_size):
                continue

            sequences.append((xb, predicted_class))
        except Exception as e:
            print(f"[오류] 실패샘플 복원 실패 → {e}")
            continue

    print(f"[로딩] {symbol}-{strategy} 실패 학습 샘플 {len(sequences)}개")
    return sequences
