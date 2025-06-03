import os
import pandas as pd

LOG_FILE = "/persistent/logs/train_log.csv"
MODEL_DIR = "/persistent/models"
ACCURACY_THRESHOLD = 0.5  # ✅ 최소 정확도 기준

def get_model_weight(model_type, strategy, symbol="ALL", max_records=100):
    """
    모델 정확도 기반 가중치 반환:
    - 최근 학습 로그에서 해당 모델의 정확도를 불러옴
    - 정확도가 낮으면 0.0 반환 (사실상 예측 제외)
    """
    try:
        if not os.path.exists(LOG_FILE):
            return 1.0  # 로그가 없으면 기본 가중치

        df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
        df = df[(df["model"] == model_type) & (df["strategy"] == strategy)]
        if symbol != "ALL":
            df = df[df["symbol"] == symbol]

        if df.empty:
            return 1.0  # 기록이 없으면 기본값

        df = df.sort_values(by="timestamp", ascending=False).head(max_records)
        accuracy = df["accuracy"].mean()

        return round(accuracy, 4) if accuracy >= ACCURACY_THRESHOLD else 0.0

    except Exception as e:
        print(f"[가중치 계산 오류] {e}")
        return 1.0  # 오류 발생 시 기본값

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
                strategy = parts[1]
                if strategy in counts:
                    counts[strategy] += 1
    except Exception as e:
        print(f"[오류] 모델 수 계산 실패: {e}")
    return counts
