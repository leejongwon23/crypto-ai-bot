import os
import pandas as pd
import json

LOG_FILE = "/persistent/logs/train_log.csv"
EVAL_RESULT = "/persistent/evaluation_result.csv"  # ✅ 평가 결과 로그
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=10):
    import os, glob, json, pandas as pd
    MODEL_DIR = "/persistent/models"

    pattern = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.meta.json") if symbol != "ALL" \
              else os.path.join(MODEL_DIR, f"*_{strategy}_{model_type}.meta.json")
    meta_files = glob.glob(pattern)

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            pt_path = meta_path.replace(".meta.json", ".pt")
            if not os.path.exists(pt_path):
                continue

            eval_files = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
            if not eval_files:
                return 1.0

            df_list = []
            for file in eval_files:
                df = pd.read_csv(file, encoding="utf-8-sig")
                df_list.append(df)
            if not df_list:
                return 1.0

            df = pd.concat(df_list, ignore_index=True)
            df = df[(df["model"] == model_type) & (df["strategy"] == strategy) &
                    (df["symbol"] == meta.get("symbol")) & (df["status"].isin(["success", "fail"]))]

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
            print(f"[get_model_weight 예외] {e}")
            continue

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
