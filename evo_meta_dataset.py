# ✅ 파일명: evo_meta_dataset.py

import pandas as pd
import numpy as np

def prepare_evo_meta_dataset(csv_path="/persistent/wrong_predictions.csv"):
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("❌ [evo_meta_dataset] 실패 예측 데이터가 비어 있음")
        return None, None

    df = df.dropna(subset=["strategy", "softmax", "expected_return", "actual_return", "predicted_class", "true_class"])
    
    # 실패율 계산용 그룹
    grouped = df.groupby("strategy")["success"].agg(["count", "sum"])
    grouped["failure_rate"] = 1 - grouped["sum"] / grouped["count"]
    failure_dict = grouped["failure_rate"].to_dict()
    
    # 입력값 구성
    X, y = [], []
    for _, row in df.iterrows():
        x = [
            float(row["softmax"]),
            float(row["expected_return"]),
            float(row["actual_return"]),
        ]
        X.append(x)
        y.append(row["strategy"])

    return np.array(X), np.array(y)
