import os
import pandas as pd
import numpy as np
from data.utils import get_kline_by_strategy, compute_features
from logger import get_feature_hash  # ✅ 중복 체크 위한 해시 필요
from failure_db import load_existing_failure_hashes
from config import get_NUM_CLASSES  # ✅ 함수 import 추가

NUM_CLASSES = get_NUM_CLASSES()  # ✅ 함수 호출 후 변수 할당

WRONG_CSV = "/persistent/wrong_predictions.csv"

def load_training_prediction_data(symbol, strategy, input_size, window, group_id=None, min_per_class=10):
    import random, os
    import numpy as np
    import pandas as pd
    from collections import Counter
    from config import FAIL_AUGMENT_RATIO, get_class_ranges, set_NUM_CLASSES
    from data.utils import get_kline_by_strategy, compute_features
    from logger import get_feature_hash
    from failure_db import load_existing_failure_hashes

    WRONG_CSV = "/persistent/wrong_predictions.csv"
    sequences = []

    # ✅ 클래스 범위 계산
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    num_classes = len(class_ranges)
    set_NUM_CLASSES(num_classes)

    df_price = get_kline_by_strategy(symbol, strategy)
    if df_price is None or df_price.empty:
        print(f"[❌ 실패] {symbol}-{strategy}: get_kline_by_strategy → 데이터 없음")
        return None, None

    df_feat = compute_features(symbol, df_price, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        print(f"[❌ 실패] {symbol}-{strategy}: compute_features → 데이터 없음")
        return None, None

    # ✅ 타임스탬프 컬럼 통일
    if "timestamp" not in df_feat.columns:
        if "datetime" in df_feat.columns:
            df_feat.rename(columns={"datetime": "timestamp"}, inplace=True)
        else:
            raise Exception("timestamp 또는 datetime 컬럼 없음")

    df_feat = df_feat.dropna().reset_index(drop=True)

    # ✅ 라벨링
    returns = df_price["close"].pct_change().fillna(0).values
    labels = []
    for r in returns:
        matched = False
        for i, rng in enumerate(class_ranges):
            if isinstance(rng, tuple) and len(rng) == 2:
                low, high = rng
                if low <= r <= high:
                    labels.append(i)
                    matched = True
                    break
        if not matched:
            labels.append(0)
    df_feat["label"] = labels[:len(df_feat)]

    used_hashes = set()
    existing_hashes = load_existing_failure_hashes()

    fail_count, normal_count = 0, 0

    # === 1. 실패 샘플 우선 수집 ===
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
                    sequences.append((xb.copy(), label))
                    fail_count += 1
        except Exception as e:
            print(f"[⚠️ 실패 로드 예외] {symbol}-{strategy}: {e}")

    # === 2. 정규 학습 샘플 수집 ===
    for i in range(window, len(df_feat)):
        try:
            window_df = df_feat.iloc[i - window:i]
            label = int(df_feat.iloc[i].get("label", -1))
            if not (0 <= label < num_classes):
                continue
            xb = window_df.drop(columns=["timestamp", "label"]).to_numpy(dtype=np.float32)
            xb = np.pad(xb, ((0, 0), (0, input_size - xb.shape[1])), mode="constant")
            if xb.shape != (window, input_size):
                continue
            h = get_feature_hash(xb[-1])
            if h in used_hashes:
                continue
            used_hashes.add(h)
            sequences.append((xb.copy(), label))
            normal_count += 1
        except Exception:
            continue

    # === 3. 클래스별 최소 샘플 보장 ===
    label_counts = Counter([s[1] for s in sequences])
    all_by_label = {cls: [] for cls in range(num_classes)}
    for xb, y in sequences:
        all_by_label[y].append(xb)

    for cls in range(num_classes):
        while len(all_by_label[cls]) < min_per_class:
            # 인접 클래스에서 샘플 가져오기
            neighbors = [c for c in [cls - 1, cls + 1] if 0 <= c < num_classes and all_by_label[c]]
            candidates = sum([all_by_label[c] for c in neighbors], [])
            if not candidates:
                # fallback: 랜덤 더미
                dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
                all_by_label[cls].append(dummy)
            else:
                xb = random.choice(candidates)
                noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                all_by_label[cls].append(xb + noise)

    # === 4. 최종 시퀀스 구성 ===
    sequences = []
    for cls, xb_list in all_by_label.items():
        for xb in xb_list:
            sequences.append((xb, cls))

    # === 5. 데이터 부족시 fallback ===
    if not sequences:
        for _ in range(FAIL_AUGMENT_RATIO * 2):
            dummy = np.random.normal(0, 1, (window, input_size)).astype(np.float32)
            random_label = random.randint(0, num_classes - 1)
            sequences.append((dummy, random_label))

    # === 최종 결과 ===
    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.int64)

    print(f"[✅ load_training_prediction_data 완료] {symbol}-{strategy} → 실패데이터 {fail_count} / 정상데이터 {normal_count} / 최종 {len(y)} (클래스별 최소 {min_per_class} 보장)")
    return X, y


