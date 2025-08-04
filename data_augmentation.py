# data_augmentation.py

import numpy as np

def add_gaussian_noise(X, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def apply_scaling(X, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return X * scale

def apply_shift(X, shift_range=(-0.1, 0.1)):
    shift = np.random.uniform(shift_range[0], shift_range[1])
    return X + shift

def apply_dropout_mask(X, dropout_prob=0.1):
    mask = np.random.binomial(1, 1-dropout_prob, X.shape)
    return X * mask

def augment_batch(X_batch):
    """
    X_batch: numpy array (batch_size, window, feature_dim)
    returns augmented X_batch
    """
    X_aug = []
    for X in X_batch:
        X1 = add_gaussian_noise(X)
        X2 = apply_scaling(X1)
        X3 = apply_shift(X2)
        X4 = apply_dropout_mask(X3)
        X_aug.append(X4)
    return np.array(X_aug, dtype=np.float32)

import numpy as np
from collections import Counter

def balance_classes(X, y, min_count=5, num_classes=None):
    import numpy as np
    from collections import Counter
    from data_augmentation import augment_batch
    import random

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("[❌ balance_classes 실패] X 또는 y 비어있음")
        raise Exception("⛔ balance_classes 중단: X 또는 y 비어있음")

    if num_classes is None:
        num_classes = int(np.max(y)) + 1

    y = y.astype(np.int64)
    mask = (y != -1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(y) == 0:
        raise Exception("⛔ balance_classes 중단: 유효 라벨 없음")

    class_counts = Counter(y)
    print(f"[📊 클래스 분포] {dict(class_counts)}")

    nsamples, nx, ny_dim = X.shape
    X_balanced, y_balanced = list(X), list(y)

    max_count = max(class_counts.values()) if class_counts else min_count
    target_count = max(min_count, int(max_count * 0.8))

    all_by_label = {cls: [] for cls in range(num_classes)}
    for xb, yb in zip(X, y):
        all_by_label[yb].append(xb)

    for cls in range(num_classes):
        existing = all_by_label.get(cls, [])
        count = len(existing)
        needed = max(0, target_count - count)

        # ✅ 기존 데이터가 조금이라도 있으면 무조건 증강
        if count > 0 and needed > 0:
            reps = np.random.choice(count, needed, replace=True)
            base = np.array([existing[i] for i in reps])
            aug = augment_batch(base)
            X_balanced.extend(aug)
            y_balanced.extend([cls] * needed)
            continue

        # 인접 클래스 복제
        if needed > 0:
            candidates = []
            for neighbor in [cls - 1, cls + 1]:
                if 0 <= neighbor < num_classes:
                    candidates += all_by_label.get(neighbor, [])
            if candidates:
                for _ in range(needed):
                    xb = random.choice(candidates)
                    noise = np.random.normal(0, 0.01, xb.shape).astype(np.float32)
                    X_balanced.append(xb + noise)
                    y_balanced.append(cls)
                continue

        # ✅ 완전 0개일 때만 더미 생성
        if count == 0 and needed > 0:
            dummy = np.random.normal(0, 1, (needed, nx, ny_dim)).astype(np.float32)
            dummy = np.clip(dummy, -3, 3)
            X_balanced.extend(dummy)
            y_balanced.extend([cls] * needed)

    combined = list(zip(X_balanced, y_balanced))
    np.random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)

    final_counts = Counter(y_shuffled)
    print(f"[📊 최종 클래스 분포] {dict(final_counts)}")

    X_final = np.array(X_shuffled, dtype=np.float32)
    y_final = np.array(y_shuffled, dtype=np.int64)

    print(f"[✅ balance_classes 완료] 총 샘플수: {len(y_final)}")
    return X_final, y_final



