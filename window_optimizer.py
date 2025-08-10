# === window_optimizer.py (FINAL) ===
import numpy as np
import pandas as pd

from data.utils import get_kline_by_strategy, compute_features
from config import get_class_ranges, get_FEATURE_INPUT_SIZE

FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

# ──────────────────────────────────────────────────────────────
# 내부 유틸: 전략별 평가 구간(시간)
# ──────────────────────────────────────────────────────────────
def _strategy_horizon_hours(strategy: str) -> int:
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)

# ──────────────────────────────────────────────────────────────
# 내부 유틸: 미래 수익률(look‑ahead) 계산
# ──────────────────────────────────────────────────────────────
def _future_returns_by_timestamp(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df is None or df.empty or "timestamp" not in df.columns:
        return np.zeros(len(df), dtype=np.float32)

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
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    labels = []
    for r in future_gains:
        idx = 0
        for i, (lo, hi) in enumerate(class_ranges):
            if lo <= r <= hi:
                idx = i
                break
        labels.append(idx)
    return np.array(labels, dtype=np.int64)

# ──────────────────────────────────────────────────────────────
# 점수 함수(가벼운 휴리스틱): 최근 피처 변동성 × 라벨 변화율
# ──────────────────────────────────────────────────────────────
def _window_score(feat_scaled: np.ndarray, labels: np.ndarray, window: int) -> float:
    if len(feat_scaled) < window or len(labels) < window:
        return -np.inf
    # 전체 피처에 대한 표준편차(최근 window 구간)
    recent_vol = float(np.std(feat_scaled[-window:], dtype=np.float32))
    # 최근 window 구간 내 라벨 변화 비율
    diffs = np.diff(labels[-window:])
    label_change = float(np.mean(diffs != 0)) if len(diffs) > 0 else 0.0
    score = recent_vol * (1.0 + label_change)
    if np.isnan(score) or np.isinf(score):
        return -np.inf
    return score

# ──────────────────────────────────────────────────────────────
# 외부 API (train.py에서 사용하는 형태)
# ──────────────────────────────────────────────────────────────
def find_best_window(symbol: str, strategy: str, window_list=None, group_id=None):
    """
    train.py에서 호출하는 시그니처.
    - look-ahead 라벨링과 동일 로직으로, 주어진 window_list 중 최적을 선택
    """
    if not window_list:
        window_list = [10, 20, 30, 40, 60]

    # 1) 데이터/피처 로드
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print(f"[find_best_window] 데이터 없음 → fallback={min(window_list)}")
        return int(min(window_list))

    feat = compute_features(symbol, df, strategy)
    if feat is None or feat.empty:
        print(f"[find_best_window] 피처 없음 → fallback={min(window_list)}")
        return int(min(window_list))

    features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
    if features_only.shape[1] < FEATURE_INPUT_SIZE:
        # 피처 패딩
        pad = FEATURE_INPUT_SIZE - features_only.shape[1]
        for i in range(pad):
            features_only[f"pad_{i}"] = 0.0
    elif features_only.shape[1] > FEATURE_INPUT_SIZE:
        features_only = features_only.iloc[:, :FEATURE_INPUT_SIZE]

    feat_scaled = features_only.to_numpy(dtype=np.float32)

    # 2) 미래 수익률 → 라벨
    gains = _future_returns_by_timestamp(df, _strategy_horizon_hours(strategy))
    labels = _label_from_future_returns(gains, symbol, strategy, group_id=group_id)

    # 3) 윈도우별 점수 계산
    best_w, best_s = int(min(window_list)), -np.inf
    for w in sorted(set(int(x) for x in window_list)):
        # 최소 길이 확보
        if len(feat_scaled) < w + 5:
            continue
        s = _window_score(feat_scaled, labels, w)
        if s > best_s:
            best_s, best_w = s, w

    if best_s == -np.inf:
        print(f"[find_best_window] 유효 윈도우 없음 → fallback={min(window_list)}")
        return int(min(window_list))

    print(f"[find_best_window] {symbol}-{strategy} -> best={best_w} (score={best_s:.6f})")
    return int(best_w)

def find_best_windows(symbol: str, strategy: str, window_list=None, group_id=None):
    """
    앙상블용: 학습 가능한 윈도우만 추려서 점수 상위 3개 반환.
    """
    if not window_list:
        window_list = [10, 20, 30, 40, 60]

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print(f"[find_best_windows] 데이터 없음 → 기본 반환 {window_list}")
        return window_list

    feat = compute_features(symbol, df, strategy)
    if feat is None or feat.empty:
        print(f"[find_best_windows] 피처 없음 → 기본 반환 {window_list}")
        return window_list

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

    scored = []
    for w in sorted(set(int(x) for x in window_list)):
        if len(feat_scaled) < w + 5:
            continue
        s = _window_score(feat_scaled, labels, w)
        if s == -np.inf:
            continue
        scored.append((w, s))

    if not scored:
        return [int(min(window_list))]

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [w for w, _ in scored[:3]]
    print(f"[find_best_windows] {symbol}-{strategy} -> {top}")
    return top
