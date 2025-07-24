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
    num_classes = len(group_classes)

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        print(f"[❌ 실패] {symbol}-{strategy}: get_kline_by_strategy → 데이터 없음")
        return None, None

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        print(f"[❌ 실패] {symbol}-{strategy}: compute_features → 데이터 없음")
        return None, None

    df_feat["timestamp"] = df_feat.get("timestamp") or df_feat.get("datetime")
    df_feat = df_feat.dropna().reset_index(drop=True)

    returns = df_price["close"].pct_change().fillna(0).values
    labels = []
    for r in returns:
        for i, (low, high) in enumerate(group_classes):
            if low <= r <= high:
                labels.append(i)
                break
        else:
            labels.append(0)
    df_feat["label"] = labels[:len(df_feat)]

    used_hashes = set()
    existing_hashes = load_existing_failure_hashes()

    ### 1. 실패 샘플 수집
    if os.path.exists(WRONG_CSV):
        try:
            df = pd.read_csv(WRONG_CSV, encoding="utf-8-sig")
            df = df[(df["symbol"] == symbol) & (df["strategy"] == strategy)]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[df["timestamp"].notna()]
            df["label"] = df.get("predicted_class", -1).astype(int)
            df = df[(df["label"] >= 0) & (df["label"] < num_classes)]

            for _, row in df.iterrows():
                entry_time = row["timestamp"]
                label = int(row["label"])
                past_window = df_feat[df_feat["timestamp"] < entry_time].tail(window)
                if past_window.empty:
                    continue
                xb = past_window.drop(columns=["timestamp", "label"]).to_numpy(dtype=np.float32)
                xb = np.pad(xb, ((window - len(xb), 0), (0, input_size - xb.shape[1])), mode="constant")
                if xb.shape != (window, input_size):
                    continue
                h = get_feature_hash(xb[-1])
                if h in used_hashes or h in existing_hashes:
                    continue
                used_hashes.add(h)
                for _ in range(FAIL_AUGMENT_RATIO * 2):
                    sequences.append((xb, label))
        except Exception as e:
            print(f"[⚠️ 실패 로드 예외] {symbol}-{strategy}: {e}")

    ### 2. 정규 학습 샘플 수집
    label_missing = []
    for i in range(window, len(df_feat)):
        try:
            window_df = df_feat.iloc[i - window:i]
            label = int(df_feat.iloc[i].get("label", -1))
            if not (0 <= label < num_classes):
                label_missing.append(label)
                continue
            xb = window_df.drop(columns=["timestamp", "label"]).to_numpy(dtype=np.float32)
            xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
            if xb.shape[0] < window:
                xb = np.pad(xb, ((window - xb.shape[0], 0), (0, 0)), mode="constant")
            if xb.shape != (window, input_size):
                continue
            h = get_feature_hash(xb[-1])
            if h in used_hashes:
                continue
            used_hashes.add(h)
            sequences.append((xb, label))
        except Exception as e:
            print(f"[❌ 정규 샘플 예외] {symbol}-{strategy}: {e}")
            continue

    ### 3. 클래스 누락시 인접 복제
    label_counts = Counter([s[1] for s in sequences])
    all_by_label = {cls: [] for cls in range(num_classes)}
    for xb, y in sequences:
        all_by_label[y].append(xb)

    for cls in range(num_classes):
        if label_counts[cls] == 0:
            print(f"[📌 클래스 {cls} 누락 → 인접 샘플 복제]")
            neighbors = [c for c in [cls-1, cls+1] if 0 <= c < num_classes and all_by_label[c]]
            candidates = sum([all_by_label[c] for c in neighbors], [])
            if not candidates:
                continue
            for _ in range(5):
                xb = random.choice(candidates)
                noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                sequences.append((xb + noise, cls))

    ### 4. 전체 부족할 경우 fallback
    if not sequences:
        print(f"[⚠️ 전체 데이터 없음] {symbol}-{strategy} → fallback 샘플 생성")
        for _ in range(FAIL_AUGMENT_RATIO * 2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            random_label = random.randint(0, num_classes - 1)
            sequences.append((dummy, random_label))

    ### 최종 결과
    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)
    print(f"[✅ load_training_prediction_data 완료] {symbol}-{strategy} → 샘플 수: {len(y)} / 클래스 분포: {dict(Counter(y))} / 누락된 라벨 수: {len(label_missing)}")

    return X, y


