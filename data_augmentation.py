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
def balance_classes(X, y, min_count=5, num_classes=21):
    import numpy as np
    from collections import Counter
    from data_augmentation import augment_batch

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("[❌ balance_classes 실패] X 또는 y 비어있음")
        raise Exception("⛔ balance_classes 중단: X 또는 y 비어있음")

    y = y.astype(np.int64)
    mask = (y != -1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(y) == 0:
        print("[❌ balance_classes 실패] 유효 라벨 없음")
        raise Exception("⛔ balance_classes 중단: 유효 라벨 없음")

    class_counts = Counter(y)
    print(f"[🔢 기존 클래스 분포] {dict(class_counts)}")

    nsamples, nx, ny_dim = X.shape
    X_balanced, y_balanced = list(X), list(y)

    max_count = max(class_counts.values()) if class_counts else min_count
    target_count = max(min_count, int(max_count * 0.8))

    for cls in range(num_classes):
        indices = [i for i, label in enumerate(y) if label == cls]
        count = len(indices)
        needed = max(0, target_count - count)

        if needed > 0:
            if count >= 1:
                try:
                    reps = np.random.choice(indices, needed, replace=True)
                    base_samples = X[reps]
                    aug_samples = augment_batch(base_samples)
                    X_balanced.extend(aug_samples)
                    y_balanced.extend([cls] * needed)
                    print(f"[✅ 클래스 {cls}] {needed}개 증강 완료")
                except Exception as e:
                    print(f"[⚠️ 클래스 {cls} 증강 실패 → noise dummy 대체] {e}")
                    dummy = np.random.normal(0, 1, (needed, nx, ny_dim)).astype(np.float32)
                    dummy = np.clip(dummy, -3, 3)
                    X_balanced.extend(dummy)
                    y_balanced.extend([cls] * needed)
            else:
                dummy = np.random.normal(0, 1, (needed, nx, ny_dim)).astype(np.float32)
                dummy = np.clip(dummy, -3, 3)
                X_balanced.extend(dummy)
                y_balanced.extend([cls] * needed)
                print(f"[🆕 클래스 {cls}] {needed}개 dummy 생성")

    combined = list(zip(X_balanced, y_balanced))
    np.random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)

    final_counts = Counter(y_shuffled)
    print(f"[📊 최종 클래스 분포] {dict(final_counts)}")

    X_final = np.array(X_shuffled, dtype=np.float32)
    y_final = np.array(y_shuffled, dtype=np.int64)

    print(f"[✅ balance_classes 완료] X.shape={X_final.shape}, y.shape={y_final.shape}, 총 샘플수: {len(y_final)}")
    return X_final, y_final

