import os
import pandas as pd
import logger

LOG_FILE = "/persistent/logs/train_log.csv"

def get_model_weight(model_type, strategy, symbol="ALL", max_records=100):
    """
    accuracy + success_rate 기반 weight 계산 (0.0 ~ 1.0)
    """
    acc_score = 1.0
    success_score = 0.5  # 기본값

    # --- accuracy 기반 계산 ---
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
            if symbol != "ALL":
                df = df[df["symbol"] == symbol]
            df = df[(df["model"] == model_type) & (df["strategy"] == strategy)]
            df = df.sort_values("timestamp", ascending=False).head(max_records)
            if len(df) > 0:
                acc_score = df["accuracy"].mean()
        except Exception as e:
            print(f"[경고] accuracy 기반 가중치 계산 실패: {e}")

    # --- success_rate 기반 계산 ---
    try:
        sr = logger.get_model_success_rate(symbol, strategy, model_type)
        if 0 <= sr <= 1:
            success_score = sr
    except Exception as e:
        print(f"[경고] success_rate 계산 실패: {e}")

    # --- 평균 가중치 반환 ---
    weight = round((acc_score + success_score) / 2, 4)
    return weight
