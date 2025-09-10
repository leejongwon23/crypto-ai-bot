# === window_optimizer.py (CV macro-F1 scoring, low-variance tiebreak, fast path) ===
import numpy as np
import pandas as pd

from data.utils import get_kline_by_strategy, compute_features
from config import get_class_ranges, get_FEATURE_INPUT_SIZE

# lightweight CV / metrics
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# tiny torch linear baseline (very fast)
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

# 최근 구간만 사용(속도 최적화)
_MAX_ROWS_FOR_SCORING = 800  # 필요 시 600~1000 사이에서 조정 가능
_MIN_SAMPLES_PER_CLASS = 2   # CV 안정성 확보용

# ──────────────────────────────────────────────────────────────
# 내부 유틸: 전략별 평가 구간(시간)
# ──────────────────────────────────────────────────────────────
def _strategy_horizon_hours(strategy: str) -> int:
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)

# ──────────────────────────────────────────────────────────────
# 내부 유틸: 미래 수익률(look-ahead) 계산 (최근 구간만)
# ──────────────────────────────────────────────────────────────
def _future_returns_by_timestamp(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df is None or df.empty or "timestamp" not in df.columns:
        return np.zeros(len(df) if df is not None else 0, dtype=np.float32)

    # 최근 구간만 사용
    df = df.tail(_MAX_ROWS_FOR_SCORING).copy()

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")

    close = df["close"].astype(float).values
    high  = (df["high"] if "high" in df.columns else df["close"]).astype(float).values

    out = np.zeros(len(df), dtype=np.float32)
    horizon = pd.Timedelta(hours=horizon_hours)

    j_start = 0
    for i in range(len(df)):
        t0 = ts.iloc[i]
        t1 = t0 + horizon
        j = max(j_start, i)
        max_h = high[i]
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > max_h:
                max_h = high[j]
            j += 1
        j_start = max(j_start, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        out[i] = float((max_h - base) / (base + 1e-12))
    return out.astype(np.float32)

# ──────────────────────────────────────────────────────────────
# 내부 유틸: 미래 수익률 → 클래스 매핑
# ──────────────────────────────────────────────────────────────
def _label_from_future_returns(future_gains: np.ndarray, symbol: str, strategy: str, group_id=None) -> np.ndarray:
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id) or []
    labels = []
    lo0 = class_ranges[0][0] if class_ranges else 0.0
    hiN = class_ranges[-1][1] if class_ranges else 0.0
    for r in future_gains:
        if not np.isfinite(r): r = lo0
        if r <= lo0: labels.append(0); continue
        if r >= hiN: labels.append(len(class_ranges) - 1 if class_ranges else 0); continue
        idx = 0
        for i, (lo, hi) in enumerate(class_ranges):
            if lo <= r <= hi:
                idx = i
                break
        labels.append(idx)
    return np.array(labels, dtype=np.int64)

# ──────────────────────────────────────────────────────────────
# 시퀀스 데이터셋 구성 (윈도우별)
# ──────────────────────────────────────────────────────────────
def _build_sequences(feat_scaled: np.ndarray, labels: np.ndarray, window: int):
    if len(feat_scaled) < window + 1:
        return None, None
    X, y = [], []
    for i in range(len(feat_scaled) - window):
        X.append(feat_scaled[i:i + window])
        yi = i + window - 1
        y.append(labels[yi] if 0 <= yi < len(labels) else 0)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y

# ──────────────────────────────────────────────────────────────
# 아주 가벼운 선형 분류기 (torch)로 CV macro-F1 계산
# 실패 시 휴리스틱으로 폴백
# ──────────────────────────────────────────────────────────────
def _cv_macro_f1_score(X: np.ndarray, y: np.ndarray, n_classes: int, folds: int = 3, epochs: int = 3) -> float:
    if (not _HAS_SKLEARN) or (not _HAS_TORCH):
        return np.nan

    # 각 클래스 표본수 확인 (CV 안정성)
    uniq, cnts = np.unique(y, return_counts=True)
    if (len(uniq) < 2) or np.min(cnts) < _MIN_SAMPLES_PER_CLASS:
        return np.nan

    # 입력을 평탄화하여 초경량 선형 분류
    Xf = X.reshape(len(X), -1)
    skf = StratifiedKFold(n_splits=min(folds, np.min(cnts)))
    f1s = []

    for tr_idx, va_idx in skf.split(Xf, y):
        Xtr = torch.tensor(Xf[tr_idx], dtype=torch.float32)
        ytr = torch.tensor(y[tr_idx], dtype=torch.long)
        Xva = torch.tensor(Xf[va_idx], dtype=torch.float32)
        yva = torch.tensor(y[va_idx], dtype=torch.long)

        model = nn.Linear(Xtr.shape[1], n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = crit(model(Xtr), ytr)
            if torch.isfinite(loss):
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(Xva), dim=1).cpu().numpy()
        f1s.append(f1_score(yva.cpu().numpy(), pred, average="macro"))

    return float(np.mean(f1s)) if f1s else np.nan

# ──────────────────────────────────────────────────────────────
# 휴리스틱 백업 점수: 최근 피처 변동성 × 라벨 변화율
# ──────────────────────────────────────────────────────────────
def _heuristic_score(feat_scaled: np.ndarray, labels: np.ndarray, window: int) -> float:
    if len(feat_scaled) < window or len(labels) < window:
        return -np.inf
    recent_vol = float(np.std(feat_scaled[-window:], dtype=np.float32))
    diffs = np.diff(labels[-window:])
    label_change = float(np.mean(diffs != 0)) if len(diffs) > 0 else 0.0
    score = recent_vol * (1.0 + label_change)
    if np.isnan(score) or np.isinf(score):
        return -np.inf
    return score

# ──────────────────────────────────────────────────────────────
# tiebreak: 최근 분산이 더 낮은 윈도우 선호
# ──────────────────────────────────────────────────────────────
def _recent_variance(feat_scaled: np.ndarray, window: int) -> float:
    if len(feat_scaled) < window:
        return np.inf
    return float(np.var(feat_scaled[-window:], dtype=np.float32))

# ──────────────────────────────────────────────────────────────
# 외부 API (train.py에서 사용하는 형태)
# ──────────────────────────────────────────────────────────────
def find_best_window(symbol: str, strategy: str, window_list=None, group_id=None):
    """
    - 점수 = Stratified K-Fold macro-F1 평균 (초경량 선형분류)
    - 동률이면 최근 분산 낮은 쪽을 선택
    - 실패/미지원 시 휴리스틱(_heuristic_score)로 폴백
    - 속도 최적화: 최근 최대 _MAX_ROWS_FOR_SCORING 행만 사용
    """
    if not window_list:
        window_list = [20, 40]

    # 1) 데이터/피처 로드 (최근 구간만)
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print(f"[find_best_window] 데이터 없음 → fallback={min(window_list)}")
        return int(min(window_list))
    df = df.tail(_MAX_ROWS_FOR_SCORING).copy()

    feat = compute_features(symbol, df, strategy)
    if feat is None or feat.empty:
        print(f"[find_best_window] 피처 없음 → fallback={min(window_list)}")
        return int(min(window_list))

    feat = feat.tail(_MAX_ROWS_FOR_SCORING).copy()
    features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
    # 입력 차원 정규화
    if features_only.shape[1] < FEATURE_INPUT_SIZE:
        pad = FEATURE_INPUT_SIZE - features_only.shape[1]
        for i in range(pad):
            features_only[f"pad_{i}"] = 0.0
    elif features_only.shape[1] > FEATURE_INPUT_SIZE:
        features_only = features_only.iloc[:, :FEATURE_INPUT_SIZE]
    feat_scaled = features_only.to_numpy(dtype=np.float32)

    # 2) 미래 수익률 → 라벨
    gains = _future_returns_by_timestamp(df, _strategy_horizon_hours(strategy))
    labels = _label_from_future_returns(gains, symbol, strategy, group_id=group_id)
    n_classes = int(np.max(labels)) + 1 if labels.size else 0

    # 3) 윈도우별 점수 계산 (CV macro-F1 → 휴리스틱 폴백)
    best_w = int(min(window_list))
    best_score = -np.inf
    best_var = np.inf

    for w in sorted(set(int(x) for x in window_list)):
        if len(feat_scaled) < w + 5:
            continue
        X, y = _build_sequences(feat_scaled, labels, w)
        if X is None or len(np.unique(y)) < 2:
            score = _heuristic_score(feat_scaled, labels, w)
        else:
            score = _cv_macro_f1_score(X, y, n_classes=n_classes, folds=3, epochs=3)
            if np.isnan(score):
                score = _heuristic_score(feat_scaled, labels, w)

        # 동률(±1e-6)이면 최근 분산 낮은 쪽 선택
        var = _recent_variance(feat_scaled, w)
        better = (score > best_score + 1e-6) or (abs(score - best_score) <= 1e-6 and var < best_var)
        if better:
            best_score, best_w, best_var = score, w, var

    if best_score == -np.inf:
        print(f"[find_best_window] 유효 윈도우 없음 → fallback={min(window_list)}")
        return int(min(window_list))

    print(f"[find_best_window] {symbol}-{strategy} -> best={best_w} (score={best_score:.6f}, var={best_var:.6f})")
    return int(best_w)

def find_best_windows(symbol: str, strategy: str, window_list=None, group_id=None):
    """
    앙상블용: 학습 가능한 윈도우만 추려서 CV macro-F1 상위 3개 반환.
    - 실패/미지원 시 휴리스틱으로 점수.
    - 속도 최적화: 최근 최대 _MAX_ROWS_FOR_SCORING 행만 사용
    """
    if not window_list:
        window_list = [20, 40]

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print(f"[find_best_windows] 데이터 없음 → 기본 반환 {window_list}")
        return [int(x) for x in window_list]
    df = df.tail(_MAX_ROWS_FOR_SCORING).copy()

    feat = compute_features(symbol, df, strategy)
    if feat is None or feat.empty:
        print(f"[find_best_windows] 피처 없음 → 기본 반환 {window_list}")
        return [int(x) for x in window_list]
    feat = feat.tail(_MAX_ROWS_FOR_SCORING).copy()

    features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
    if features_only.shape[1] < FEATURE_INPUT_SIZE:
        pad = FEATURE_INPUT_SIZE - features_only.shape[1]
        for i in range(pad):
            features_only[f"pad_{i}"] = 0.0
    elif features_only.shape[1] > FEATURE_INPUT_SIZE:
        features_only = features_only.iloc[:, :FEATURE_INPUT_SIZE]
    feat_scaled = features_only.to_numpy(dtype=np.float32)

    gains = _future_returns_by_timestamp(df, _strategy_horizon_hours(strategy))
    labels = _label_from_future_returns(gains, symbol, strategy, group_id=group_id)
    n_classes = int(np.max(labels)) + 1 if labels.size else 0

    scored = []
    for w in sorted(set(int(x) for x in window_list)):
        if len(feat_scaled) < w + 5:
            continue
        X, y = _build_sequences(feat_scaled, labels, w)
        if X is None or len(np.unique(y)) < 2:
            s = _heuristic_score(feat_scaled, labels, w)
        else:
            s = _cv_macro_f1_score(X, y, n_classes=n_classes, folds=3, epochs=3)
            if np.isnan(s):
                s = _heuristic_score(feat_scaled, labels, w)
        if s == -np.inf:
            continue
        scored.append((w, s, _recent_variance(feat_scaled, w)))

    if not scored:
        return [int(min(window_list))]

    # 1차: 점수 내림차순, 2차: 최근분산 오름차순
    scored.sort(key=lambda x: (-x[1], x[2]))
    top = [int(w) for w, _, _ in scored[:3]]
    print(f"[find_best_windows] {symbol}-{strategy} -> {top}")
    return top
