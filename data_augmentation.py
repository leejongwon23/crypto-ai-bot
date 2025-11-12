# data_augmentation.py — 증강/클래스균형 + (옵션) 패턴유사도 보류까지 "한 파일"로 통합
from __future__ import annotations
import numpy as np
from collections import Counter
from typing import Tuple, Optional, Dict, List

# (옵션) torch 샘플러 지원  # ✅ 추가
try:
    import torch
    from torch.utils.data import WeightedRandomSampler
except Exception:
    torch = None
    WeightedRandomSampler = None

# ─────────────────────────────────────────────────────────────
# (옵션) 데이터 접근: predict 단계 유사도 보류용
# ─────────────────────────────────────────────────────────────
try:
    from data.utils import get_kline_by_strategy
except Exception:
    try:
        from utils import get_kline_by_strategy  # 루트 폴백
    except Exception:
        get_kline_by_strategy = None  # 없으면 보류 기능은 자동 비활성

# ─────────────────────────────────────────────────────────────
# 기본 증강 유틸
# ─────────────────────────────────────────────────────────────
def add_gaussian_noise(X: np.ndarray, mean: float = 0.0, std: float = 0.01,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    noise = rng.normal(mean, std, X.shape).astype(np.float32)
    return (X.astype(np.float32) + noise)

def apply_scaling(X: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05),
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    scale = float(rng.uniform(*scale_range))
    return (X.astype(np.float32) * scale)

def apply_shift(X: np.ndarray, shift_range: Tuple[float, float] = (-0.02, 0.02),
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    shift = float(rng.uniform(*shift_range))
    return (X.astype(np.float32) + shift)

def apply_dropout_mask(X: np.ndarray, dropout_prob: float = 0.05,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random
    mask = rng.binomial(1, 1.0 - float(dropout_prob), X.shape).astype(np.float32)
    return (X.astype(np.float32) * mask)

# ─────────────────────────────────────────────────────────────
# 배치 증강
# ─────────────────────────────────────────────────────────────
def augment_batch(X_batch: np.ndarray,
                  add_noise_prob: float = 0.9,
                  scale_prob: float = 0.6,
                  shift_prob: float = 0.5,
                  dropout_prob: float = 0.3,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    X_batch: (batch, window, feat) 또는 (window, feat)
    반환: float32 배열
    """
    if X_batch is None:
        return np.array([], dtype=np.float32)

    rng_local = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    arr = np.array(X_batch, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    outs: List[np.ndarray] = []
    for x in arr:
        xa = x.copy()
        try:
            if rng_local.random() < add_noise_prob:
                xa = add_gaussian_noise(xa, std=0.01, rng=rng_local)
            if rng_local.random() < scale_prob:
                xa = apply_scaling(xa, (0.95, 1.05), rng_local)
            if rng_local.random() < shift_prob:
                xa = apply_shift(xa, (-0.02, 0.02), rng_local)
            if rng_local.random() < dropout_prob:
                xa = apply_dropout_mask(xa, 0.05, rng_local)
        except Exception:
            xa = x.copy()
        outs.append(xa.astype(np.float32))

    return np.stack(outs, axis=0)

# ─────────────────────────────────────────────────────────────
# 소수 클래스 복제 + 미세 노이즈
# ─────────────────────────────────────────────────────────────
def _dup_with_noise(X: np.ndarray, times: int, noise_level: float = 0.005) -> np.ndarray:
    if times <= 1:
        return X.astype(np.float32)
    X_rep = np.repeat(X.astype(np.float32), times, axis=0)
    if noise_level > 0:
        std = np.std(X, axis=0)
        std = np.where(np.isfinite(std), std, 0.0)
        std = np.clip(std, 1e-6, None)
        noise = np.random.normal(0.0, noise_level, size=X_rep.shape) * std
        X_rep = (X_rep + noise.astype(np.float32)).astype(np.float32)
    return X_rep

# ─────────────────────────────────────────────────────────────
# 클래스 불균형 보정 (안정 캡 포함)
# ─────────────────────────────────────────────────────────────
def balance_classes(X: np.ndarray,
                    y: np.ndarray,
                    min_count: int = 5,
                    num_classes: Optional[int] = None,
                    target_ratio: float = 0.8,
                    rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    - 소수 클래스만 적당히 오버샘플
    - 과도 증폭 방지: 상위 75퍼센타일 캡 + 배수 상한(2~6배)
    - 원본 분포를 크게 훼손하지 않도록 경미 변형 위주
    - ⚠️ '라벨 병합/축소'는 하지 않음 (YOPO 철학 준수)
    """
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        raise ValueError("balance_classes 중단: X 또는 y 비어있음")

    rng_local = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)

    # 기대 형태: (N, window, feat)
    if X.ndim != 3 or y.ndim != 1 or len(X) != len(y):
        return X, y

    if num_classes is None:
        num_classes = int(np.max(y)) + 1 if len(y) > 0 else 0
    if num_classes <= 1:
        return X, y

    cc = Counter(y.tolist())
    print(f"[📊 클래스 분포] {dict(cc)}")

    X_parts = [X]
    y_parts = [y]

    nz = np.array([c for c in cc.values() if c > 0], dtype=int)
    if nz.size == 0:
        return X, y
    cap = max(int(np.percentile(nz, 75)), int(np.mean(nz)), 32)

    idx_by_cls = {c: np.where(y == c)[0] for c in range(num_classes)}
    for cls in range(num_classes):
        idx = idx_by_cls.get(cls, np.array([], dtype=int))
        c = int(len(idx))
        if c <= 0 or c >= cap:
            continue
        need = cap - c
        times = int(np.ceil((c + need) / max(c, 1)))
        times = max(2, min(times, 6))

        X_cls = X[idx]
        X_aug = _dup_with_noise(X_cls, times=times, noise_level=0.005)
        take = min(max(0, len(X_aug) - c), need) if len(X_aug) > c else min(len(X_aug), need)
        if take > 0:
            base = X_aug[:take]
            try:
                base = augment_batch(base, rng=rng_local)
            except Exception:
                pass
            X_parts.append(base.astype(np.float32))
            y_parts.append(np.full((take,), cls, dtype=np.int64))

    X_out = np.concatenate(X_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)

    perm = rng_local.permutation(len(y_out))
    X_out = X_out[perm].astype(np.float32)
    y_out = y_out[perm].astype(np.int64)

    print(f"[📊 최종 클래스 분포] {dict(Counter(y_out.tolist()))}")
    print(f"[✅ balance_classes 완료] 총 샘플수: {len(y_out)} (캡={cap})")
    return X_out, y_out

# ─────────────────────────────────────────────────────────────
# ✅ 학습용 클래스 가중치 & 샘플러 (라벨 병합 없이 보정)
# ─────────────────────────────────────────────────────────────
def compute_class_weights(y: np.ndarray,
                          num_classes: Optional[int] = None,
                          method: str = "effective",
                          beta: float = 0.999) -> np.ndarray:
    """
    method:
      - "inverse":  weight_c = 1 / count_c
      - "effective": weight_c = (1 - beta) / (1 - beta**count_c)  (Cui et al., 2019)
    반환: shape=(num_classes,) float32
    """
    y = np.asarray(y).astype(np.int64)
    if y.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts <= 0] = 1.0

    if method == "inverse":
        w = 1.0 / counts
    else:  # effective
        beta = float(beta)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)

def make_sample_weights(y: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """
    각 샘플별 weight = class_weights[y_i]
    """
    y = np.asarray(y).astype(np.int64)
    w = np.asarray(class_weights, dtype=np.float32)
    if y.size == 0 or w.size == 0:
        return np.zeros((0,), dtype=np.float32)
    w = w / (w.mean() + 1e-12)
    return w[y]

def make_weighted_sampler(y: np.ndarray,
                          class_weights: Optional[np.ndarray] = None,
                          method: str = "effective",
                          beta: float = 0.999,
                          replacement: bool = True):
    """
    PyTorch WeightedRandomSampler 생성 (가능할 때만).
    - torch 미설치/불가 시 None 반환
    """
    if torch is None or WeightedRandomSampler is None:
        return None
    y = np.asarray(y).astype(np.int64)
    if y.size == 0:
        return None
    if class_weights is None:
        class_weights = compute_class_weights(y, method=method, beta=beta)
    sample_w = make_sample_weights(y, class_weights)
    tens = torch.as_tensor(sample_w, dtype=torch.float32)
    return WeightedRandomSampler(weights=tens, num_samples=len(tens), replacement=replacement)

# ─────────────────────────────────────────────────────────────
# (옵션) 코사인 유사도 기반 보류 Guard — predict에서 필요 시 사용
# ─────────────────────────────────────────────────────────────
def _to_returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    if close.size < 3:
        return np.zeros((0,), dtype=np.float64)
    r = np.diff(close) / (close[:-1] + 1e-12)
    r = np.tanh(r * 5.0)
    r = r - r.mean()
    std = r.std() + 1e-12
    return (r / std)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def _rolling_template(x: np.ndarray, w: int, stride: int = 8) -> np.ndarray:
    vs = []
    if x.size < w:
        return np.zeros((0, w), dtype=np.float64)
    for i in range(0, x.size - w + 1, stride):
        vs.append(x[i:i+w])
    if not vs:
        return np.zeros((0, w), dtype=np.float64)
    return np.stack(vs, axis=0)

def should_abstain_by_similarity(symbol: str, strategy: str, window: int = 60, thr: float = 0.10) -> bool:
    """
    최근 window 길이 반환율 패턴과 과거 템플릿의 평균 코사인 유사도가 thr 미만이면 보류(True).
    get_kline_by_strategy가 없거나 데이터 부족이면 False(보류 아님).
    """
    if get_kline_by_strategy is None:
        return False
    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or "close" not in df.columns or len(df) < (window * 4):
            return False
        close = df["close"].to_numpy(dtype=np.float64)
        r = _to_returns(close)
        if r.size < (window * 4):
            return False

        cur = r[-window:]
        hist = r[:-(window*3)]
        tmpl = _rolling_template(hist, w=window, stride=max(4, window // 8))
        if tmpl.shape[0] == 0:
            return False

        sims = np.array([_cosine(cur, t) for t in tmpl], dtype=np.float64)
        sim_mean = float(np.nanmean(sims)) if sims.size else 0.0
        return (sim_mean < float(thr))
    except Exception:
        return False
