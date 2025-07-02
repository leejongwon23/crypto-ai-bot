import os
import pandas as pd
import json
import glob

MODEL_DIR = "/persistent/models"
EVAL_RESULT = "/persistent/evaluation_result.csv"

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=10, input_size=None):
    pattern = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.meta.json") if symbol != "ALL" \
              else os.path.join(MODEL_DIR, f"*_{strategy}_{model_type}.meta.json")
    meta_files = glob.glob(pattern)

    if not meta_files:
        print(f"[⚠️ meta 파일 없음] {pattern} → weight=0")
        return 0.0

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if input_size is not None and meta.get("input_size") != input_size:
                print(f"[⚠️ input_size 불일치] {meta.get('input_size')} vs {input_size} → weight=0")
                continue  # ✅ continue로 수정하여 다른 파일 탐색

            pt_path = meta_path.replace(".meta.json", ".pt")
            if not os.path.exists(pt_path):
                print(f"[⚠️ 모델 파일 없음] {pt_path} → weight=0")
                continue  # ✅ continue로 수정하여 다른 파일 탐색

            eval_files = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
            if not eval_files:
                print("[INFO] 평가 파일 없음 → cold-start weight=0.2")
                return 0.2

            df_list = []
            for file in eval_files:
                try:
                    df = pd.read_csv(file, encoding="utf-8-sig")
                    df_list.append(df)
                except Exception as e:
                    print(f"[⚠️ 평가 파일 읽기 실패] {file} → {e}")
                    continue

            if not df_list:
                print("[INFO] 평가 데이터 없음 → cold-start weight=0.2")
                return 0.2

            df = pd.concat(df_list, ignore_index=True)
            df = df[(df["model"] == model_type) & (df["strategy"] == strategy) &
                    (df["symbol"] == meta.get("symbol")) & (df["status"].isin(["success", "fail"]))]

            if len(df) < min_samples:
                print(f"[INFO] 평가 샘플 부족(len={len(df)}) → cold-start weight=0.2")
                return 0.2

            success_rate = len(df[df["status"] == "success"]) / len(df)
            if success_rate >= 0.7:
                return 1.0
            elif success_rate < 0.3:
                return 0.0
            else:
                w = max(0.0, round((success_rate - 0.3) / (0.7 - 0.3), 4))
                print(f"[INFO] success_rate={success_rate:.4f}, weight={w}")
                return w

        except Exception as e:
            print(f"[get_model_weight 예외] {e}")
            continue

    # ✅ 모든 meta 파일 검사 후 조건 충족 weight 없으면 cold-start 반환
    print("[INFO] 조건 충족 모델 없음 → cold-start weight=0.2")
    return 0.2


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
