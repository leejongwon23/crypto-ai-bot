import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

    grouped = df.groupby("predicted_class")
    sequences = []

    for cls, group in grouped:
        if len(group) < min_samples:
            continue

        for _, row in group.iterrows():
            try:
                entry_time = row["timestamp"]
                predicted_class = int(row["predicted_class"])
                rate = float(row.get("rate", 0.0))

                seq = []
                for i in range(window):
                    t = entry_time - timedelta(minutes=(window - i) * 5)
                    # 랜덤 값 대신 수치 그대로 넣거나 0.0 처리
                    vec = np.array([rate] * input_size)
                    seq.append(vec)

                xb = np.array(seq)
                sequences.append((xb, predicted_class))
            except Exception as e:
                print(f"[오류] 실패샘플 복원 실패 → {e}")
                continue

    print(f"[로딩] {symbol}-{strategy} 실패 학습 샘플 {len(sequences)}개")
    return sequences
