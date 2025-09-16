# data_augmentation.py — 안정적이고 초보 친화적으로 정리된 버전
import numpy as np
import random
from collections import Counter
from typing import Tuple, Optional

# ---------- 기본 증강 유틸 ----------
def add_gaussian_noise(X: np.ndarray, mean: float = 0.0, std: float = 0.01, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    noise = rng.normal(mean, std, X.shape).astype(np.float32)
    return (X.astype(np.float32) + noise)

def apply_scaling(X: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1), rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    return (X.astype(np.float32) * scale)

def apply_shift(X: np.ndarray, shift_range: Tuple[float, float] = (-0.1, 0.1), rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    shift = float(rng.uniform(shift_range[0], shift_range[1]))
    return (X.astype(np.float32) + shift)

def apply_dropout_mask(X: np.ndarray, dropout_prob: float = 0.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    mask = rng.binomial(1, 1.0 - float(dropout_prob), X.shape).astype(np.float32)
    return (X.astype(np.float32) * mask)

# ---------- 배치 증강 ----------
def augment_batch(X_batch: np.ndarray,
                  add_noise_prob: float = 0.9,
                  scale_prob: float = 0.7,
                  shift_prob: float = 0.5,
                  dropout_prob: float = 0.3,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    X_batch: numpy array (batch_size, window, feature_dim) or (window, feature_dim)
    returns augmented X_batch (dtype=float32)
    """
    if X_batch is None:
        return np.array([], dtype=np.float32)

    rng_local = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

    arr = np.array(X_batch, dtype=np.float32)
    # if single sample passed, make it 1D batch
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    out_list = []
    for x in arr:
        x_aug = x.copy()
        # Probabilistic application of augmentations (combination possible)
        try:
            if rng_local.random() < add_noise_prob:
                x_aug = add_gaussian_noise(x_aug, std=0.01, rng=rng_local)
            if rng_local.random() < scale_prob:
                x_aug = apply_scaling(x_aug, scale_range=(0.95, 1.05), rng=rng_local)
            if rng_local.random() < shift_prob:
                x_aug = apply_shift(x_aug, shift_range=(-0.02, 0.02), rng=rng_local)
            if rng_local.random() < dropout_prob:
                x_aug = apply_dropout_mask(x_aug, dropout_prob=0.05, rng=rng_local)
        except Exception:
            # If any op fails, fall back to original x
            x_aug = x.copy()
        out_list.append(x_aug.astype(np.float32))

    out = np.stack(out_list, axis=0)
    # if original was single sample, return shape (1, window, feat)
    return out

# ---------- 클래스 불균형 보정 ----------
def balance_classes(X: np.ndarray,
                    y: np.ndarray,
                    min_count: int = 5,
                    num_classes: Optional[int] = None,
                    target_ratio: float = 0.8,
                    rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    안전한 클래스 균형 함수.
    - X: ndarray (n_samples, window, feature_dim)
    - y: ndarray (n_samples,) integer labels
    - returns (X_balanced, y_balanced)
    """

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        raise ValueError("balance_classes 중단: X 또는 y 비어있음")

    rng_local = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

    if num_classes is None:
        num_classes = int(np.max(y)) + 1 if len(y) > 0 else 0

    y = np.asarray(y).astype(np.int64)

    # mask out invalid labels (-1 or non-finite)
    valid_mask = (y != -1) & np.isfinite(y)
    X = np.asarray(X, dtype=np.float32)[valid_mask]
    y = y[valid_mask]

    if len(y) == 0:
        raise ValueError("balance_classes 중단: 유효 라벨 없음")

    class_counts = Counter(y.tolist())
    print(f"[📊 클래스 분포] {dict(class_counts)}")

    n_samples, win_len, feat_dim = X.shape
    X_balanced = list(X)
    y_balanced = list(y)

    # target: either at least min_count, or target_ratio * current max class count
    max_count = max(class_counts.values()) if class_counts else 0
    target_count = max(min_count, int(max(1, max_count) * float(target_ratio)))

    # create dictionary of samples per class
    samples_by_label = {cls: [] for cls in range(num_classes)}
    for xb, yb in zip(X, y):
        if 0 <= int(yb) < num_classes:
            samples_by_label[int(yb)].append(xb)

    for cls in range(num_classes):
        existing = samples_by_label.get(cls, [])
        count = len(existing)
        needed = max(0, target_count - count)
        if needed == 0:
            continue

        # 1) 증강이 가능한 경우 (복제 + augment)
        if count > 0:
            try:
                # choose base indices with replacement
                idxs = rng_local.integers(0, count, size=needed)
                base = np.stack([existing[i] for i in idxs], axis=0)
                aug = augment_batch(base, rng=rng_local)
                X_balanced.extend(aug)
                y_balanced.extend([cls] * len(aug))
                continue
            except Exception as e:
                print(f"[⚠️ 증강 실패] 클래스 {cls} → {e}")

        # 2) 인접 클래스에서 샘플 가져와 소량 노이즈 추가
        candidates = []
        for neighbor in (cls - 1, cls + 1):
            if 0 <= neighbor < num_classes:
                candidates.extend(samples_by_label.get(neighbor, []))
        if candidates:
            for _ in range(needed):
                xb = candidates[rng_local.integers(0, len(candidates))]
                noise = rng_local.normal(0, 0.01, size=xb.shape).astype(np.float32)
                X_balanced.append((xb + noise).astype(np.float32))
                y_balanced.append(cls)
            continue

        # 3) 전혀 샘플 없는 클래스는 데이터 통계 기반 더미 생성 (평균/표준편차 사용)
        # fallback: use global mean/std from dataset if possible
        try:
            global_mean = np.mean(X, axis=(0, 1), keepdims=False)
            global_std = np.std(X, axis=(0, 1), keepdims=False) + 1e-6
            dummy = rng_local.normal(loc=global_mean, scale=global_std, size=(needed, win_len, feat_dim)).astype(np.float32)
        except Exception:
            dummy = rng_local.normal(0, 1.0, size=(needed, win_len, feat_dim)).astype(np.float32)
            dummy = np.clip(dummy, -3, 3).astype(np.float32)

        X_balanced.extend(dummy)
        y_balanced.extend([cls] * needed)

    # shuffle final set
    X_arr = np.array(X_balanced, dtype=np.float32)
    y_arr = np.array(y_balanced, dtype=np.int64)
    perm = rng_local.permutation(len(y_arr))
    X_final = X_arr[perm]
    y_final = y_arr[perm]

    final_counts = Counter(y_final.tolist())
    print(f"[📊 최종 클래스 분포] {dict(final_counts)}")
    print(f"[✅ balance_classes 완료] 총 샘플수: {len(y_final)}")

    return X_final, y_final
