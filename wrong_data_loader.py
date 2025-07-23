import os
import pandas as pd
import numpy as np
from data.utils import get_kline_by_strategy, compute_features
from logger import get_feature_hash  # ✅ 중복 체크 위한 해시 필요
from failure_db import load_existing_failure_hashes
from config import get_NUM_CLASSES  # ✅ 함수 import 추가

NUM_CLASSES = get_NUM_CLASSES()  # ✅ 함수 호출 후 변수 할당

WRONG_CSV = "/persistent/wrong_predictions.csv"

def load_training_prediction_data(symbol, strategy, input_size, window, group_id=None):
    import random, os
    import numpy as np
    import pandas as pd
    from collections import Counter
    from config import FAIL_AUGMENT_RATIO, get_class_groups
    from data.utils import get_kline_by_strategy, compute_features
    from logger import get_feature_hash
    from failure_db import load_existing_failure_hashes

    WRONG_CSV = "/persistent/wrong_predictions.csv"
    sequences = []

    # ✅ 그룹별 클래스 제한
    class_groups = get_class_groups()
    group_classes = class_groups[group_id] if group_id is not None else list(range(sum(len(g) for g in class_groups)))

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        return None, None

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        return None, None

    if "timestamp" not in df_feat.columns:
        df_feat["timestamp"] = df_feat.get("datetime")
    df_feat = df_feat.dropna().reset_index(drop=True)

    used_hashes = set()
    existing_hashes = load_existing_failure_hashes()

    if os.path.exists(WRONG_CSV):
        try:
            df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig")
            df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"].notna()]

            if "label" not in df.columns:
                if "predicted_class" in df.columns:
                    df["label"] = pd.to_numeric(df["predicted_class"], errors="coerce").fillna(-1).astype(int)
                else:
                    df["label"] = -1

            df = df[df["label"] >= 0]

            for _, row in df.iterrows():
                try:
                    entry_time = row["timestamp"]
                    label = int(row["label"])
                    if label not in group_classes:
                        continue

                    past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
                    if len(past_window) < window:
                        continue

                    xb = past_window.drop(columns=["timestamp"]).to_numpy(dtype=np.float32)
                    if xb.shape[1] < input_size:
                        xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant", constant_values=0)
                    if xb.shape != (window, input_size):
                        continue

                    h = get_feature_hash(xb[-1])
                    if h in used_hashes or h in existing_hashes:
                        continue
                    used_hashes.add(h)

                    for _ in range(FAIL_AUGMENT_RATIO * 2):
                        sequences.append((xb, label))
                except:
                    continue
        except Exception as e:
            print(f"[⚠️ 실패기록 파싱 오류] {symbol}-{strategy} → {e}")

    label_counts = Counter([s[1] for s in sequences])
    for cls in group_classes:
        if label_counts[cls] == 0:
            print(f"[📌 클래스 {cls} 누락 → dummy 생성]")
            for _ in range(5):
                dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
                sequences.append((dummy, cls))

    if not sequences:
        print(f"[⚠️ 데이터 없음] {symbol}-{strategy} → fallback 샘플 생성")
        for _ in range(FAIL_AUGMENT_RATIO * 2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            random_label = np.random.choice(group_classes)
            sequences.append((dummy, random_label))

    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)
    print(f"[✅ load_training_prediction_data 완료] {symbol}-{strategy}-g{group_id} → 샘플 수: {len(y)} / 클래스 수: {len(set(y))}")
    return X, y
