import os
import pandas as pd

LOG_FILE = "/persistent/logs/train_log.csv"
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", max_records=100):
    """
    모델 가중치 계산은 사용하지 않음.
    모든 모델은 개별 예측을 하고, 점수 판단은 수익률 기준으로 이뤄짐.
    따라서 항상 기본값 1.0을 반환함.
    """
    return 1.0

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
