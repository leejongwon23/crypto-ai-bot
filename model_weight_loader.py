import os
import pandas as pd
import json

LOG_FILE = "/persistent/logs/train_log.csv"
EVAL_RESULT = "/persistent/evaluation_result.csv"  # ✅ 평가 결과 로그
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=10):
    import glob, json, os
    import pandas as pd

    MODEL_DIR = "/persistent/models"

    pattern = os.path.join(MODEL_DIR, f"*_{strategy}_*.meta.json") if symbol == "ALL" \
              else os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*.meta.json")
    meta_files = glob.glob(pattern)

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            m = meta.get("model", "").strip()
            s = meta.get("strategy", "").strip()
            sy = meta.get("symbol", "").strip()

            if m != model_type or s != strategy or (symbol != "ALL" and sy != symbol):
                continue

            pt_path = meta_path.replace(".meta.json", ".pt")
            if not os.path.exists(pt_path):
                print(f"[❌ PT 없음] {pt_path}")
                return 1.0  # ✅ PT 없으면도 기본 가중치 1.0 반환

            eval_files = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
            if not eval_files:
                print("[⚠️ 평가 로그 없음] 기본 가중치 1.0 반환")
                return 1.0

            df_list = []
            for file in eval_files:
                try:
                    df = pd.read_csv(file, encoding="utf-8-sig")
                    df_list.append(df)
                except:
                    continue

            if not df_list:
                print("[⚠️ 평가 데이터 없음] 기본 가중치 1.0 반환")
                return 1.0

            df = pd.concat(df_list, ignore_index=True)
            df = df[(df["model"] == model_type) &
                    (df["strategy"] == strategy) &
                    ((df["symbol"] == sy) if symbol != "ALL" else True) &
                    (df["status"].isin(["success", "fail"]))]

            if len(df) < min_samples:
                print(f"[⚠️ 샘플 부족] {len(df)}개 → 기본 가중치 1.0 반환")
                return 1.0

            success_rate = len(df[df["status"] == "success"]) / len(df)
            print(f"[✅ SUCCESS RATE] {symbol}-{strategy}-{model_type}: {success_rate:.2%}")

            # ✅ 어떤 성공률이든 기본 1.0 반환
            return 1.0

        except Exception as e:
            print(f"[❌ 처리 실패] {meta_path} → {e}")
            continue

    print(f"[❌ 매칭 실패] {symbol}-{strategy}-{model_type} 해당 모델 없음 → 기본 가중치 1.0 반환")
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
