import os
import pandas as pd
import logger

LOG_FILE = "/persistent/logs/train_log.csv"
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", max_records=100):
    """
    모델별 앙상블 가중치 계산 (정확도 + 성공률 + 예측이력 품질 고려):
    - accuracy: 학습 정확도 (최근 로그 기준)
    - success_rate: 예측 성공률 (logger 기준)
    - failure_penalty: 실패율 반영
    - weight = 정규화된 복합 점수
    """

    acc_score = 0.5    # 학습 정확도 기반
    success_score = 0.5  # 예측 성공률 기반
    failure_penalty = 1.0  # 실패율 감소 계수
    weight = 0.5       # 최종 가중치 기본값

    # --- 학습 정확도 기반 계산 ---
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
            if symbol != "ALL":
                df = df[df["symbol"] == symbol]
            df = df[(df["model"] == model_type) & (df["strategy"] == strategy)]
            df = df.sort_values("timestamp", ascending=False).head(max_records)
            if len(df) > 0:
                acc_score = float(df["accuracy"].mean())
        except Exception as e:
            print(f"[경고] accuracy 기반 가중치 계산 실패: {e}")

    # --- 예측 성공률 기반 계산 ---
    try:
        sr = logger.get_model_success_rate(symbol, strategy, model_type)
        if 0 <= sr <= 1:
            success_score = sr
    except Exception as e:
        print(f"[경고] success_rate 계산 실패: {e}")

    # --- 실패율 패널티 계산 ---
    try:
        fail_rate = logger.get_strategy_fail_rate(symbol, strategy)
        failure_penalty = 1.0 - min(fail_rate, 0.5)  # 실패율 최대 50%만 감안
    except Exception as e:
        print(f"[경고] 실패율 계산 실패: {e}")

    # --- 최종 가중치 계산 ---
    try:
        combined = (acc_score * 0.4) + (success_score * 0.5)
        weight = round(combined * failure_penalty, 4)
    except:
        weight = 0.5

    print(f"[가중치 계산] {symbol}-{strategy}-{model_type} → acc: {round(acc_score,4)} / sr: {round(success_score,4)} / fail_penalty: {round(failure_penalty,4)} → weight: {weight}")
    return weight


def model_exists(symbol, strategy):
    """
    해당 symbol-strategy 조합의 모델이 최소 1개라도 존재하는지 확인
    """
    try:
        for file in os.listdir(MODEL_DIR):
            if file.startswith(f"{symbol}_{strategy}_") and file.endswith(".pt"):
                return True
    except Exception as e:
        print(f"[오류] 모델 존재 확인 실패: {e}")
    return False
