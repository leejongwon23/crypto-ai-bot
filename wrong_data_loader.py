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

    # ✅ 실패 기록 기반 샘플
    if os.path.exists(WRONG_CSV):
        try:
            df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig")
            df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"].notna()]

            if "label" not in df.columns:
                df["label"] = df.get("predicted_class", -1).astype(int)

            df = df[df["label"] >= 0]

            for _, row in df.iterrows():
                entry_time = row["timestamp"]
                label = int(row["label"])
                if label not in group_classes:
                    continue

                past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
                if len(past_window) < window:
                    continue

                xb = past_window.drop(columns=["timestamp"]).to_numpy(dtype=np.float32)
                if xb.shape[1] < input_size:
                    xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
                if xb.shape != (window, input_size):
                    continue

                h = get_feature_hash(xb[-1])
                if h in used_hashes or h in existing_hashes:
                    continue
                used_hashes.add(h)

                for _ in range(FAIL_AUGMENT_RATIO * 2):
                    sequences.append((xb, label))
        except:
            pass

    # ✅ 정규 샘플
    for i in range(window, len(df_feat)):
        try:
            window_df = df_feat.iloc[i - window:i]
            label = int(df_feat.iloc[i].get("class", -1))
            if label not in group_classes:
                continue

            xb = window_df.drop(columns=["timestamp"]).to_numpy(dtype=np.float32)
            if xb.shape[1] < input_size:
                xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
            if xb.shape != (window, input_size):
                continue

            h = get_feature_hash(xb[-1])
            if h in used_hashes:
                continue
            used_hashes.add(h)

            sequences.append((xb, label))
        except:
            continue

    # ✅ 부족 클래스는 인접 클래스에서 유사 샘플을 복제해 채움
    label_counts = Counter([s[1] for s in sequences])
    all_by_label = {cls: [] for cls in group_classes}
    for xb, y in sequences:
        all_by_label[y].append(xb)

    for cls in group_classes:
        if label_counts[cls] == 0:
            print(f"[📌 클래스 {cls} 누락 → 인접 샘플 복제]")
            neighbors = [cls - 1, cls + 1]
            candidates = []
            for n in neighbors:
                if n in all_by_label:
                    candidates += all_by_label[n]
            if not candidates:
                continue
            for _ in range(5):
                xb = random.choice(candidates)
                noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                sequences.append((xb + noise, cls))

    # ✅ fallback
    if not sequences:
        print(f"[⚠️ 전체 데이터 없음] {symbol}-{strategy} → fallback 샘플 생성")
        for _ in range(FAIL_AUGMENT_RATIO * 2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            random_label = random.choice(group_classes)
            sequences.append((dummy, random_label))

    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)
    counts = dict(Counter(y))
    print(f"[✅ 데이터 로드 완료] {symbol}-{strategy}-g{group_id} → 총: {len(y)}개 / 분포: {counts}")
    return X, y
