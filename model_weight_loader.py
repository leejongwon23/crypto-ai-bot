import os
import pandas as pd

LOG_FILE = "/persistent/logs/train_log.csv"
EVAL_RESULT = "/persistent/evaluation_result.csv"  # ✅ 평가 결과 로그
MODEL_DIR = "/persistent/models"

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=10):
    import glob, json

    MODEL_DIR = "/persistent/models"
    meta_path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.meta.json"
    pt_path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.pt"

    # ✅ 모델 파일 존재 확인
    if not os.path.exists(meta_path) or not os.path.exists(pt_path):
        print(f"[❌ 누락] 모델 또는 메타 파일 없음 → {symbol}-{strategy}-{model_type}")
        return 0.0

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # ✅ 메타 키 유효성 체크 (설계상 핵심 3개 키)
        m = meta.get("model", "").strip()
        s = meta.get("strategy", "").strip()
        sy = meta.get("symbol", "").strip()

        if not m or not s or not sy:
            print(f"[❌ 메타 누락] 'model', 'strategy', 'symbol' 중 일부 없음 → {meta_path}")
            return 0.0

        if m != model_type or s != strategy or sy != symbol:
            print(f"[⚠️ 불일치] 메타 정보 불일치 → {meta_path}")
            return 0.0

    except Exception as e:
        print(f"[❌ 메타 로딩 실패] {meta_path} → {e}")
        return 0.0

    try:
        eval_files = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
        if not eval_files:
            return 1.0

        import pandas as pd
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
        print(f"[❌ 평가기반 가중치 계산 실패] → {e}")
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

