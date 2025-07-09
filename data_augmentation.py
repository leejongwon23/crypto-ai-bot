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

def balance_classes(X, y, min_count=20, num_classes=21):
    import numpy as np
    from collections import Counter

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("[❌ balance_classes 실패] X 또는 y 비어있음")
        return X, y

    y = y.astype(np.int64)
    mask = (y != -1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(y) == 0:
        raise Exception("[❌ balance_classes 실패] 라벨 제거 후 샘플 없음")

    class_counts = Counter(y)
    print(f"[🔢 기존 클래스 분포] {dict(class_counts)}")

    nsamples, nx, ny = X.shape
    X_balanced, y_balanced = list(X), list(y)

    max_count = max(class_counts.values()) if class_counts else min_count
    target_count = max(min_count, int(max_count * 0.8))

    for cls in range(num_classes):
        indices = [i for i, label in enumerate(y) if label == cls]
        count = len(indices)
        needed = max(0, target_count - count)

        if needed > 0:
            if count >= 1:
                reps = np.random.choice(indices, needed, replace=True)
                base_samples = X[reps]

                # ✅ Gaussian noise
                noisy_samples = base_samples + np.random.normal(0, 0.05, base_samples.shape).astype(np.float32)

                # ✅ Scaling augmentation
                scale = np.random.uniform(0.9, 1.1, size=(needed, 1, 1)).astype(np.float32)
                scaled_samples = noisy_samples * scale

                # ✅ Mixup + Time masking
                mixup_samples = scaled_samples.copy()
                for i in range(len(mixup_samples)):
                    j = np.random.randint(len(X))
                    lam = np.random.beta(0.2, 0.2)
                    mixup_samples[i] = lam * mixup_samples[i] + (1 - lam) * X[j]

                    t = np.random.randint(0, nx)
                    mixup_samples[i][t] = 0.0

                # ✅ NaN, Inf check
                if np.any(np.isnan(mixup_samples)) or np.any(np.isinf(mixup_samples)):
                    print(f"[⚠️ 경고] 클래스 {cls} augmentation 중 NaN 또는 Inf 발생 → 제거")
                    mixup_samples = np.nan_to_num(mixup_samples, nan=0.0, posinf=1e6, neginf=-1e6)

                X_balanced.extend(mixup_samples)
                y_balanced.extend([cls]*needed)
                print(f"[✅ 클래스 {cls}] {needed}개 추가 완료")

            else:
                print(f"[스킵] 클래스 {cls} → 샘플 없음, noise sample 생성 생략")

    combined = list(zip(X_balanced, y_balanced))
    np.random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)

    final_counts = Counter(y_shuffled)
    print(f"[📊 최종 클래스 분포] {dict(final_counts)}")
    print(f"[✅ balance_classes 완료] 최종 샘플수: {len(y_shuffled)}")

    return np.array(X_shuffled), np.array(y_shuffled, dtype=np.int64)

