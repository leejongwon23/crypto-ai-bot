import os
import pandas as pd

LOG_FILE = "/persistent/logs/train_log.csv"
EVAL_RESULT = "/persistent/evaluation_result.csv"  # ✅ 평가 결과 로그
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=10):
    """
    평가 로그를 날짜별로 읽고 모델 성능 기반 가중치 반환 (0.0 ~ 1.0)
    ※ 단, 모델이 실제 존재하면 최소 weight=1.0 반환
    """
    import glob, os
    import pandas as pd

    MODEL_DIR = "/persistent/models"
    model_base = f"{symbol}_{strategy}_{model_type}"
    model_path = os.path.join(MODEL_DIR, f"{model_base}.pt")
    meta_path = os.path.join(MODEL_DIR, f"{model_base}.meta.json")

    try:
        eval_files = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
        if not eval_files:
            return 1.0  # 평가 로그 없으면 기본값 허용

        df_list = []
        for file in eval_files:
            try:
                df = pd.read_csv(file, encoding="utf-8-sig")
                df_list.append(df)
            except:
                continue

        if not df_list:
            return 1.0

        df = pd.concat(df_list, ignore_index=True)
        df = df[(df["model"] == model_type) &
                (df["strategy"] == strategy) &
                (df["symbol"] == symbol) &
                (df["status"].isin(["success", "fail"]))]

        if len(df) < min_samples:
            return 1.0

        success_rate = len(df[df["status"] == "success"]) / len(df)

        if success_rate >= 0.65:
            return 1.0
        elif success_rate < 0.4:
            return 0.0
        else:
            return round((success_rate - 0.4) / (0.65 - 0.4), 4)

    except Exception as e:
        print(f"[오류] get_model_weight 실패: {e}")
        # ✅ 예외 발생 시라도 모델 파일이 존재하면 weight=1.0 반환
        if os.path.exists(model_path) and os.path.exists(meta_path):
            return 1.0
        return 0.0


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

