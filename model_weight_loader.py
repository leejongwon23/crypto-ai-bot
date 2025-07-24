import os
import pandas as pd
import json
import glob
import torch
import time

MODEL_DIR = "/persistent/models"
EVAL_RESULT = "/persistent/evaluation_result.csv"

# ✅ 모델 캐시 (메모리 TTL 캐싱)
_model_cache = {}
_model_cache_ttl = {}

def load_model_cached(pt_path, ttl_sec=600):
    """
    ✅ 모델을 캐싱하여 로드 (TTL 적용)
    :param pt_path: 모델 파일 경로
    :param ttl_sec: 캐시 유지 시간 (초)
    :return: torch 모델 or None
    """
    now = time.time()

    # 캐시 HIT + 유효 TTL
    if pt_path in _model_cache and now - _model_cache_ttl.get(pt_path, 0) < ttl_sec:
        print(f"[캐시 HIT] {pt_path}")
        return _model_cache[pt_path]

    # 파일 존재 여부
    if not os.path.exists(pt_path):
        print(f"[❌ load_model_cached] 파일 없음: {pt_path}")
        return None

    try:
        model = torch.load(pt_path, map_location=torch.device("cpu"))
        model.eval()
        _model_cache[pt_path] = model
        _model_cache_ttl[pt_path] = now
        print(f"[✅ 모델 로드 완료] {pt_path}")
        return model
    except Exception as e:
        print(f"[❌ load_model_cached 오류] {pt_path} → {e}")
        return None

def get_model_weight(model_type, strategy, symbol="ALL", min_samples=3, input_size=None):
    import os, glob, json
    import pandas as pd

    MODEL_DIR = "/persistent/models"
    pattern = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.meta.json") if symbol != "ALL" \
              else os.path.join(MODEL_DIR, f"*_{strategy}_{model_type}.meta.json")
    meta_files = glob.glob(pattern)

    if not meta_files:
        print(f"[⚠️ meta 파일 없음] {pattern} → fallback weight=0.2 (모델 없음)")
        return 0.2  # ✅ 모델 자체가 없을 경우 fallback weight 허용

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # ✅ input_size mismatch fallback weight
            if input_size is not None and meta.get("input_size") != input_size:
                print(f"[⚠️ input_size 불일치] meta={meta.get('input_size')} vs input={input_size} → fallback weight=0.2")
                return 0.2

            pt_path = meta_path.replace(".meta.json", ".pt")
            if not os.path.exists(pt_path):
                print(f"[⚠️ 모델 파일 없음] {pt_path} → fallback weight=0.2")
                return 0.2

            eval_files = sorted(glob.glob("/persistent/logs/evaluation_*.csv"))
            if not eval_files:
                print("[INFO] 평가 파일 없음 → cold-start fallback weight=0.2")
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
                print("[INFO] 평가 데이터 없음 → cold-start fallback weight=0.2")
                return 0.2

            df = pd.concat(df_list, ignore_index=True)
            df = df[(df["model"] == model_type) & (df["strategy"] == strategy) &
                    (df["symbol"] == meta.get("symbol")) & (df["status"].isin(["success", "fail"]))]

            sample_len = len(df)
            if sample_len < min_samples:
                print(f"[INFO] 평가 샘플 부족 (len={sample_len} < {min_samples}) → fallback weight=0.2")
                return 0.2

            success_rate = len(df[df["status"] == "success"]) / sample_len
            if success_rate >= 0.7:
                print(f"[INFO] success_rate={success_rate:.4f} → weight=1.0")
                return 1.0
            elif success_rate < 0.3:
                print(f"[INFO] success_rate={success_rate:.4f} → weight=0.0 (fallback 구조로만 사용)")
                return 0.0
            else:
                w = max(0.0, round((success_rate - 0.3) / (0.7 - 0.3), 4))
                print(f"[INFO] success_rate={success_rate:.4f}, weight={w}")
                return w

        except Exception as e:
            print(f"[get_model_weight 예외] {e}")
            continue

    print("[INFO] 조건 충족 모델 없음 → fallback weight=0.2")
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

def get_class_return_range(class_id, method="quantile", data_path="/persistent/prediction_log.csv"):
    """
    ✅ 주어진 클래스 ID의 수익률 범위를 반환
    :param class_id: 예측된 클래스 번호
    :param method: 분포 방식 ("equal" or "quantile")
    :param data_path: 예측 로그 CSV 경로
    :return: (min_return, max_return)
    """
    import numpy as np
    import pandas as pd
    from config import get_NUM_CLASSES

    num_classes = get_NUM_CLASSES()

    try:
        if method == "equal":
            step = 2.0 / num_classes  # -1.0 ~ +1.0 기준
            lower = -1.0 + class_id * step
            upper = lower + step
            return (lower, upper)

        elif method == "quantile":
            df = pd.read_csv(data_path, encoding="utf-8-sig")
            returns = df["return"].dropna().values
            if len(returns) < num_classes:
                return get_class_return_range(class_id, method="equal")

            quantiles = np.quantile(returns, np.linspace(0, 1, num_classes + 1))
            return (quantiles[class_id], quantiles[class_id + 1])

        else:
            return get_class_return_range(class_id, method="equal")

    except Exception as e:
        print(f"[❌ get_class_return_range 오류] class_id={class_id} → {e}")
        return (-1.0, 1.0)
