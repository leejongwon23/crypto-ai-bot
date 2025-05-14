import os
import pandas as pd

LOG_FILE = "/persistent/logs/train_log.csv"

def get_model_weight(model_type, strategy, max_records=100):
    if not os.path.exists(LOG_FILE):
        return 1.0  # 로그 없으면 기본값

    try:
        df = pd.read_csv(LOG_FILE)
        df = df[(df["model"] == model_type) & (df["strategy"] == strategy)]
        df = df.sort_values("timestamp", ascending=False).head(max_records)
        if len(df) == 0:
            return 1.0
        acc = df["accuracy"].mean()
        return round(acc, 4)
    except Exception as e:
        print(f"[경고] 가중치 계산 실패: {e}")
        return 1.0
