import os
import pandas as pd

LOG_FILE = "/persistent/logs/train_log.csv"
EVAL_RESULT = "/persistent/evaluation_result.csv"  # ✅ 평가 결과 로그
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=10):
    import os, json, glob, pandas as pd

    model_dir = "/persistent/models"
    meta_path = f"{model_dir}/{symbol}_{strategy}_{model_type}.meta.json"
    pt_path = f"{model_dir}/{symbol}_{strategy}_{model_type}.pt"

    if not os.path.exists(meta_path) or not os.path.exists(pt_path):
        return 0.0

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model") != model_type or meta.get("symbol") != symbol or meta.get("strategy") != strategy:
            return 0.0
    except:
        return 0.0

    try:
        df = pd.read_csv("/persistent/evaluation_result.csv", encoding="utf-8-sig")
        df = df[(df["model"] == model_type) & (df["strategy"] == strategy) & (df["symbol"] == symbol)]
        df = df[df["status"].isin(["success", "fail"])]

        if len(df) < min_samples:
            return 1.0

        success_rate = len(df[df["status"] == "success"]) / len(df)

        if success_rate >= 0.65:
            return 1.0
        elif success_rate < 0.4:
            return 0.0
        else:
            return round((success_rate - 0.4) / 0.25, 4)
    except:
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

