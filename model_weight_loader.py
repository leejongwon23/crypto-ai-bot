import os
import pandas as pd

LOG_FILE = "/persistent/logs/train_log.csv"
EVAL_RESULT = "/persistent/evaluation_result.csv"  # ✅ 평가 결과 로그
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=10):
    """
    예측 성공률 기반 가중치 반환 (0.0 ~ 1.0 사이)

    - 65% 이상 성공률: 1.0
    - 40~65% 사이: 선형 스케일링
    - 40% 미만: 0.0
    """
    if not os.path.exists(EVAL_RESULT):
        return 1.0

    try:
        df = pd.read_csv(EVAL_RESULT, encoding="utf-8-sig")
        df = df[(df["model"] == model_type) &
                (df["strategy"] == strategy) &
                (df["symbol"] == symbol) &
                (df["status"].isin(["success", "fail"]))]

        if len(df) < min_samples:
            return 1.0  # 샘플 부족 시 기본값

        success_rate = len(df[df["status"] == "success"]) / len(df)

        if success_rate >= 0.65:
            return 1.0
        elif success_rate < 0.4:
            return 0.0
        else:
            # 40~65% 사이: 0.0~1.0 선형 스케일
            return round((success_rate - 0.4) / (0.65 - 0.4), 4)

    except Exception as e:
        print(f"[오류] get_model_weight 실패: {e}")
        return 1.0


def model_exists(symbol, strategy):
    try:
        for file in os.listdir(MODEL_DIR):
            if file.startswith(f"{symbol}_{strategy}_") and file.endswith(".pt"):
                return True
    except Exception as e:
        print(f"[오류] 모델 존재 확인 실패: {e}")
    return False


def count_models_per_strategy():
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

