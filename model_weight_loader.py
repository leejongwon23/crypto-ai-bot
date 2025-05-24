import os
import pandas as pd
import logger

LOG_FILE = "/persistent/logs/train_log.csv"
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", max_records=100):
    """
    모델별 앙상블 가중치 계산:
    - accuracy: 학습 정확도
    - success_rate: 예측 성공률
    - fail_rate: 실패율
    - weight = 정규화된 점수 × 실패율 보정
    """
    acc_score = 0.5
    success_score = 0.5
    failure_penalty = 1.0
    weight = 0.5

    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
            if symbol != "ALL":
                df = df[df["symbol"] == symbol]
            df = df[(df["model"] == model_type) & (df["strategy"] == strategy)]
            df = df.sort_values("timestamp", ascending=False).head(max_records)
            if not df.empty:
                acc_score = float(df["accuracy"].mean())
        except Exception as e:
            print(f"[경고] accuracy 계산 실패: {e}")

    try:
        sr = logger.get_model_success_rate(symbol, strategy, model_type)
        if 0 <= sr <= 1:
            success_score = sr
    except Exception as e:
        print(f"[경고] success_rate 계산 실패: {e}")

    try:
        fail_rate = logger.get_strategy_fail_rate(symbol, strategy)
        failure_penalty = 1.0 - min(fail_rate, 0.5)
        failure_penalty = round(max(failure_penalty, 0.5), 4)
    except Exception as e:
        print(f"[경고] 실패율 계산 실패: {e}")

    try:
        combined = (acc_score * 0.4) + (success_score * 0.6)
        weight = round(combined * failure_penalty, 4)
    except:
        weight = 0.5

    print(f"[가중치 계산] {symbol}-{strategy}-{model_type} → acc: {round(acc_score,4)} / sr: {round(success_score,4)} / fail_penalty: {round(failure_penalty,4)} → weight: {weight}")
    return weight

def model_exists(symbol, strategy):
    """
    해당 symbol-strategy 조합에 대해 저장된 모델이 있는지 확인
    """
    try:
        for file in os.listdir(MODEL_DIR):
            if file.startswith(f"{symbol}_{strategy}_") and file.endswith(".pt"):
                return True
    except Exception as e:
        print(f"[오류] 모델 존재 확인 실패: {e}")
    return False

def count_models_per_strategy():
    """
    여포 헬스용: 전략별 모델 수 집계
    Returns:
        dict: {"단기": 60, "중기": 60, "장기": 60} 형태
    """
    counts = {"단기": 0, "중기": 0, "장기": 0}
    try:
        for file in os.listdir(MODEL_DIR):
            if not file.endswith(".pt"):
                continue
            parts = file.split("_")
            if len(parts) >= 3:
                strategy = parts[1]  # 파일명 형식: symbol_strategy_model.pt
                if strategy in counts:
                    counts[strategy] += 1
    except Exception as e:
        print(f"[오류] 모델 수 계산 실패: {e}")
    return counts
