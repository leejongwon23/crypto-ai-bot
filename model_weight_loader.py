import os
import pandas as pd
import logger

LOG_FILE = "/persistent/logs/train_log.csv"

def get_model_weight(model_type, strategy, symbol="ALL", max_records=100):
    """
    모델별 앙상블 가중치 계산:
    - 최근 accuracy 평균 + success rate 평균 → 0.0 ~ 1.0 스케일로 반환
    - accuracy: 학습 성능 기반
    - success_rate: 실제 예측 결과 기반
    """

    acc_score = 1.0    # 학습 정확도 기반
    success_score = 0.5  # 예측 성공률 기반
    weight = 0.75       # 기본값

    # --- 학습 정확도 기반 계산 ---
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

    # --- 성공률 기반 계산 ---
    try:
        sr = logger.get_model_success_rate(symbol, strategy, model_type)
        if 0 <= sr <= 1:
            success_score = sr
    except Exception as e:
        print(f"[경고] success_rate 계산 실패: {e}")

    # --- 최종 가중치 산출 (동등 평균) ---
    try:
        weight = round((acc_score + success_score) / 2, 4)
    except:
        weight = 0.75  # fallback

    print(f"[가중치 계산] {symbol}-{strategy}-{model_type} → acc: {round(acc_score,4)} / sr: {round(success_score,4)} → weight: {weight}")
    return weight
