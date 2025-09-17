# === window_optimizer.py (CV macro-F1 scoring, variance-penalized, time-guarded) ===
import os
import time
import numpy as np
import pandas as pd

from data.utils import get_kline_by_strategy, compute_features
from config import get_class_ranges, get_FEATURE_INPUT_SIZE

# optional cache (존재 시만 사용)
try:
    from data.utils import CacheManager as DataCacheManager  # noqa
    _HAS_DCACHE = True
except Exception:
    _HAS_DCACHE = False

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

# ──────────────────────────────────────────────────────────────
# 결정적 시드(동일 조건 재현)
# ──────────────────────────────────────────────────────────────
def _set_global_seed():
    try:
        s = int(os.getenv("GLOBAL_SEED", "20240101"))
        np.random.seed(s)
        if _HAS_TORCH:
            import random
            random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass

_set_global_seed()

FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

# 최근 구간만 사용(속도 최적화)
_MAX_ROWS_FOR_SCORING = int(os.getenv("WINOPT_MAX_ROWS", "800"))  # 600~1000 권장
_MIN_SAMPLES_PER_CLASS = 2   # CV 안정성 확보용

# 점수 가중치(분산 패널티)
_VAR_PENALTY = float(os.getenv("WINOPT_VAR_PENALTY", "0.05"))  # 0이면 패널티 없음

# 시간 가드
_TIME_BUDGET_SEC = float(os.getenv("WINOPT_TIMEOUT_SEC", "12"))  # 윈도우 탐색 전체 제한
_FIT_EPOCHS = int(os.getenv("WINOPT_EPOCHS", "3"))
_CV_FOLDS = int(os.getenv("WINOPT_FOLDS", "3"))

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
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")

    close = pd.to_numeric(df["close"], errors="coerce").fillna(method="ffill").fillna(method="bfill").astype(float).values
    high  = pd.to_numeric(df["high"] if "high" in df.columns else df["close"], errors="coerce").fillna(method="ffill").fillna(method="bfill").astype(float).values

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
        if not np.isfinite(r):
            r = lo0
        if r <= lo0:
            labels.append(0)
            continue
        if r >= hiN:
            labels.append(len(class_ranges) - 1 if class_ranges else 0)
            continue
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
def _cv_macro_f1_score(X: np.ndarray, y: np.ndarray, n_classes: int, folds: int = 3, epochs: int = 3, time_guard=None) -> float:
    if (not _HAS_SKLEARN) or (not _HAS_TORCH):
        return np.nan

    # 각 클래스 표본수 확인 (CV 안정성)
    uniq, cnts = np.unique(y, return_counts=True)
    if (len(uniq) < 2) or np.min(cnts) < _MIN_SAMPLES_PER_CLASS:
        return np.nan

    # 입력을 평탄화하여 초경량 선형 분류
    Xf = X.reshape(len(X), -1)
    k = int(min(folds, np.max([2, np.min(cnts)])))
    if k < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(os.getenv("GLOBAL_SEED", "20240101")))
    f1s = []

    for tr_idx, va_idx in skf.split(Xf, y):
        if time_guard and (time.time() - time_guard["t0"] > time_guard["budget"]):
            # 시간 초과 시 즉시 중단 → 현재까지 평균 반환
            return float(np.mean(f1s)) if f1s else np.nan

        Xtr = torch.tensor(Xf[tr_idx], dtype=torch.float32)
        ytr = torch.tensor(y[tr_idx], dtype=torch.long)
        Xva = torch.tensor(Xf[va_idx], dtype=torch.float32)
        yva = torch.tensor(y[va_idx], dtype=torch.long)

        model = nn.Linear(Xtr.shape[1], n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            if time_guard and (time.time() - time_guard["t0"] > time_guard["budget"]):
                break
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
    recent = feat_scaled[-window:]
    # NaN/inf 방지
    recent = np.nan_to_num(recent, nan=0.0, posinf=0.0, neginf=0.0)
    recent_vol = float(np.std(recent, dtype=np.float32))
    diffs = np.diff(labels[-window:])
    label_change = float(np.mean(diffs != 0)) if len(diffs) > 0 else 0.0
    score = recent_vol * (1.0 + label_change)
    if np.isnan(score) or np.isinf(score):
        return -np.inf
    return score

# ──────────────────────────────────────────────────────────────
# tiebreak/penalty: 최근 분산 계산
# ──────────────────────────────────────────────────────────────
def _recent_variance(feat_scaled: np.ndarray, window: int) -> float:
    if len(feat_scaled) < window:
        return np.inf
    recent = feat_scaled[-window:]
    recent = np.nan_to_num(recent, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.var(recent, dtype=np.float32))

def _apply_variance_penalty(score: float, var: float) -> float:
    # 점수에 분산 패널티 적용(낮은 분산 선호)
    if not np.isfinite(score):
        return -np.inf
    if _VAR_PENALTY <= 0:
        return score
    return float(score - _VAR_PENALTY * np.log1p(max(var, 0.0)))

# ──────────────────────────────────────────────────────────────
# 외부 API (train.py에서 사용하는 형태)
# ──────────────────────────────────────────────────────────────
def _normalize_features_df(feat: pd.DataFrame) -> np.ndarray:
    """FEATURE_INPUT_SIZE에 맞춰 패딩/슬라이스, NaN/inf 정리."""
    features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
    # NaN/inf 정리
    features_only = features_only.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 입력 차원 정규화
    if features_only.shape[1] < FEATURE_INPUT_SIZE:
        pad = FEATURE_INPUT_SIZE - features_only.shape[1]
        for i in range(pad):
            features_only[f"pad_{i}"] = 0.0
    elif features_only.shape[1] > FEATURE_INPUT_SIZE:
        features_only = features_only.iloc[:, :FEATURE_INPUT_SIZE]
    return features_only.to_numpy(dtype=np.float32)

def find_best_window(symbol: str, strategy: str, window_list=None, group_id=None):
    """
    - 점수 = Stratified K-Fold macro-F1 평균 (초경량 선형분류) - 분산 패널티 적용
    - 동률이면 최근 분산 낮은 쪽을 선택
    - 실패/미지원/시간초과 시 휴리스틱(_heuristic_score)로 폴백
    - 속도 최적화: 최근 최대 _MAX_ROWS_FOR_SCORING 행만 사용
    """
    t0 = time.time()
    if not window_list:
        window_list = [20, 40, 60]

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
    feat_scaled = _normalize_features_df(feat)

    # 2) 미래 수익률 → 라벨
    gains = _future_returns_by_timestamp(df, _strategy_horizon_hours(strategy))
    labels = _label_from_future_returns(gains, symbol, strategy, group_id=group_id)
    if labels.size == 0:
        print(f"[find_best_window] 라벨 없음 → fallback={min(window_list)}")
        return int(min(window_list))
    n_classes = int(np.max(labels)) + 1

    # 3) 윈도우별 점수 계산 (CV macro-F1 → 휴리스틱 폴백)
    best_w = int(min(window_list))
    best_score = -np.inf
    best_var = np.inf

    for w in sorted(set(int(x) for x in window_list)):
        if len(feat_scaled) < w + 5:
            continue

        # 시간 초과 가드
        if time.time() - t0 > _TIME_BUDGET_SEC:
            print(f"[find_best_window] 시간 초과 → 조기종료, 현재 best={best_w}")
            break

        X, y = _build_sequences(feat_scaled, labels, w)
        if X is None or len(np.unique(y)) < 2:
            score = _heuristic_score(feat_scaled, labels, w)
        else:
            score = _cv_macro_f1_score(
                X, y, n_classes=n_classes,
                folds=_CV_FOLDS, epochs=_FIT_EPOCHS,
                time_guard={"t0": t0, "budget": _TIME_BUDGET_SEC}
            )
            if np.isnan(score):
                score = _heuristic_score(feat_scaled, labels, w)

        var = _recent_variance(feat_scaled, w)
        adj = _apply_variance_penalty(score, var)

        # 동률(±1e-6)이면 최근 분산 낮은 쪽 선택
        better = (adj > best_score + 1e-6) or (abs(adj - best_score) <= 1e-6 and var < best_var)
        if better:
            best_score, best_w, best_var = adj, w, var

    if best_score == -np.inf:
        print(f"[find_best_window] 유효 윈도우 없음 → fallback={min(window_list)}")
        return int(min(window_list))

    print(f"[find_best_window] {symbol}-{strategy} -> best={best_w} (score={best_score:.6f}, var={best_var:.6f})")
    return int(best_w)

def find_best_windows(symbol: str, strategy: str, window_list=None, group_id=None):
    """
    앙상블용: 학습 가능한 윈도우만 추려서 (분산 패널티 적용) 상위 3개 반환.
    - 실패/미지원/시간초과 시 휴리스틱으로 점수.
    - 속도 최적화: 최근 최대 _MAX_ROWS_FOR_SCORING 행만 사용
    """
    t0 = time.time()
    if not window_list:
        window_list = [20, 40, 60]

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
    feat_scaled = _normalize_features_df(feat)

    gains = _future_returns_by_timestamp(df, _strategy_horizon_hours(strategy))
    labels = _label_from_future_returns(gains, symbol, strategy, group_id=group_id)
    if labels.size == 0:
        return [int(min(window_list))]
    n_classes = int(np.max(labels)) + 1

    scored = []
    for w in sorted(set(int(x) for x in window_list)):
        if len(feat_scaled) < w + 5:
            continue

        if time.time() - t0 > _TIME_BUDGET_SEC:
            print(f"[find_best_windows] 시간 초과 → 조기종료")
            break

        X, y = _build_sequences(feat_scaled, labels, w)
        if X is None or len(np.unique(y)) < 2:
            s = _heuristic_score(feat_scaled, labels, w)
        else:
            s = _cv_macro_f1_score(
                X, y, n_classes=n_classes,
                folds=_CV_FOLDS, epochs=_FIT_EPOCHS,
                time_guard={"t0": t0, "budget": _TIME_BUDGET_SEC}
            )
            if np.isnan(s):
                s = _heuristic_score(feat_scaled, labels, w)
        if s == -np.inf:
            continue
        var = _recent_variance(feat_scaled, w)
        scored.append((w, _apply_variance_penalty(s, var), var))

    if not scored:
        return [int(min(window_list))]

    # 1차: 조정점수 내림차순, 2차: 최근분산 오름차순
    scored.sort(key=lambda x: (-x[1], x[2]))
    top = [int(w) for w, _, _ in scored[:3]]
    print(f"[find_best_windows] {symbol}-{strategy} -> {top}")
    return top

__all__ = ["find_best_window", "find_best_windows"]
