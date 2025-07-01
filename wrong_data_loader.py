import os
import pandas as pd
import numpy as np
from data.utils import get_kline_by_strategy, compute_features
from logger import get_feature_hash  # ✅ 중복 체크 위한 해시 필요
from failure_db import load_existing_failure_hashes

WRONG_CSV = "/persistent/wrong_predictions.csv"

def load_training_prediction_data(symbol, strategy, input_size, window):
    if not os.path.exists(WRONG_CSV):
        return []

    try:
        df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig")
        df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # ✅ label 컬럼 처리: 없으면 predicted_class로 대체 후 int 변환
        if "label" not in df.columns and "predicted_class" in df.columns:
            df["label"] = df["predicted_class"]

        df = df[df["label"].notna()]
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df[df["label"].notna()]
        df["label"] = df["label"].astype(int)

        df = df.dropna(subset=["timestamp", "label"])

    except Exception as e:
        print(f"[불러오기 오류] {symbol}-{strategy} → {e}")
        return []

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        print(f"[스킵] {symbol}-{strategy} → 시세데이터 없음")
        return []

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        print(f"[스킵] {symbol}-{strategy} → 피처 생성 실패 또는 NaN 존재")
        return []

    if "timestamp" not in df_feat.columns:
        df_feat["timestamp"] = df_feat.get("datetime")
    df_feat = df_feat.dropna().reset_index(drop=True)

    existing_hashes = load_existing_failure_hashes()
    used_hashes = set()

    sequences = []
    for _, row in df.iterrows():
        try:
            entry_time = row["timestamp"]
            label = row["label"]

            entry_time = pd.to_datetime(entry_time).tz_localize("Asia/Seoul") if entry_time.tzinfo is None else entry_time

            past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
            if len(past_window) < window:
                continue

            xb = past_window.drop(columns=["timestamp"]).to_numpy(dtype=np.float32)
            if xb.shape != (window, input_size):
                continue

            h = get_feature_hash(xb[-1])
            if h in used_hashes or h in existing_hashes:
                continue
            used_hashes.add(h)

            sequences.append((xb, label))

        except Exception as e:
            print(f"[예외] {symbol}-{strategy} 실패샘플 처리 오류 → {e}")
            continue

    # ✅ fallback: 실패 데이터 없으면 빈 샘플 추가
    if not sequences:
        print(f"[INFO] {symbol}-{strategy} 실패 데이터 없음 → fallback zero sample 추가")
        zero_sample = np.zeros((window, input_size), dtype=np.float32)
        sequences.append((zero_sample, -1))

    return sequences

