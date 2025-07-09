import os
import pandas as pd
import numpy as np
from data.utils import get_kline_by_strategy, compute_features
from logger import get_feature_hash  # ✅ 중복 체크 위한 해시 필요
from failure_db import load_existing_failure_hashes

WRONG_CSV = "/persistent/wrong_predictions.csv"

def load_training_prediction_data(symbol, strategy, window, input_size):
    import random
    from config import FAIL_AUGMENT_RATIO  # ✅ 실패 복사 비율 파라미터 import

    if not os.path.exists(WRONG_CSV):
        print(f"[INFO] {symbol}-{strategy} 실패학습 파일 없음 → 스킵")
        return None, None

    try:
        df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig")
        df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if "label" not in df.columns:
            if "predicted_class" in df.columns:
                df["label"] = df["predicted_class"]
            else:
                return None, None

        df = df[df["label"].notna()]
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
        df = df.dropna(subset=["timestamp", "label"])

    except Exception as e:
        print(f"[불러오기 오류] {symbol}-{strategy} → {type(e).__name__}: {e}")
        return None, None

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        return None, None

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        return None, None

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

            # ✅ input_size fallback pad 처리
            if xb.shape[1] < input_size:
                pad_cols = input_size - xb.shape[1]
                xb = np.pad(xb, ((0,0),(0,pad_cols)), mode="constant", constant_values=0)
                print(f"[info] load_training_prediction_data pad 적용: {xb.shape}")

            if xb.shape != (window, input_size):
                continue

            h = get_feature_hash(xb[-1])
            if h in used_hashes or h in existing_hashes:
                continue
            used_hashes.add(h)

            # ✅ 실패 데이터 oversampling (FAIL_AUGMENT_RATIO배 추가)
            for _ in range(FAIL_AUGMENT_RATIO):
                sequences.append((xb, label))

            # ✅ 예측실패(-1) 라벨 augmentation
            if label == -1:
                random_label = random.randint(0, 20)
                noise_xb = xb + np.random.normal(0, 0.05, xb.shape).astype(np.float32)
                sequences.append((noise_xb, random_label))

        except Exception as e:
            print(f"[예외] {symbol}-{strategy} 실패샘플 처리 오류 → {type(e).__name__}: {e}")
            continue

    if not sequences:
        print(f"[INFO] {symbol}-{strategy} 실패 데이터 없음 → fallback noise sample 추가")
        noise_sample = np.random.normal(loc=0.0, scale=1.0, size=(window, input_size)).astype(np.float32)
        sequences.append((noise_sample, -1))

    # ✅ return X, y 형태로 수정
    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)
    return X, y
