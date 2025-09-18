# === window_optimizer.py (KST, '< t1', labels.py와 완전 일치) ===
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

_MAX_ROWS_FOR_SCORING = int(os.getenv("WINOPT_MAX_ROWS", "800"))
_MIN_SAMPLES_PER_CLASS = 2

_VAR_PENALTY = float(os.getenv("WINOPT_VAR_PENALTY", "0.05"))

_TIME_BUDGET_SEC = float(os.getenv("WINOPT_TIMEOUT_SEC", "12"))
_FIT_EPOCHS = int(os.getenv("WINOPT_EPOCHS", "3"))
_CV_FOLDS = int(os.getenv("WINOPT_FOLDS", "3"))

def _strategy_horizon_hours(strategy: str) -> int:
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)

# ── (1) 라벨 산식과 완전 동일: KST + 종가 기준 + 창경계 '< t1' ──
def _future_returns_by_timestamp(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if df is None or df.empty or "timestamp" not in df.columns:
        return np.zeros(len(df) if df is not None else 0, dtype=np.float32)

    df = df.tail(_MAX_ROWS_FOR_SCORING).copy()

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")

    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().astype(float).values

    out = np.zeros(len(df), dtype=np.float32)
    H = pd.Timedelta(hours=int(horizon_hours))

    j = 0
    n = len(df)
    for i in range(n):
        t1 = ts.iloc[i] + H
        j = max(j, i)
        while j < n and ts.iloc[j] < t1:  # ✅ '< t1' (labels.py와 동일)
            j += 1
        tgt_idx = max(i, min(j - 1, n - 1))  # 마지막 '< t1'
        ref = close[i] if close[i] != 0 else (close[i] + 1e-6)
        tgt = close[tgt_idx]
        out[i] = float((tgt - ref) / (ref + 1e-12))
    return out.astype(np.float32)

# ── (2) 구간 매핑 규칙도 labels.py와 동일: [lo, hi), 마지막만 [lo, hi] ──
def _label_from_future_returns(future_gains: np.ndarray, symbol: str, strategy: str, group_id=None) -> np.ndarray:
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id) or []
    if not class_ranges:
        return np.zeros_like(future_gains, dtype=np.int64)

    n = len(class_ranges)
    labels = np.zeros(len(future_gains), dtype=np.int64)
    for idx, g in enumerate(future_gains):
        val = float(g) if np.isfinite(g) else class_ranges[0][0]
        # 앞 구간들 [lo, hi)
        found = False
        for k, (lo, hi) in enumerate(class_ranges[:-1]):
            if (val >= lo) and (val < hi):
                labels[idx] = k
                found = True
                break
        if not found:
            lo_last, hi_last = class_ranges[-1]
            labels[idx] = n - 1 if val >= lo_last else 0
    return labels

def _build_sequences(feat_scaled: np.ndarray, labels: np.ndarray, window: int):
    if len(feat_scaled) < window + 1:
        return None, None
    X, y = [], []
    for i in range(len(feat_scaled) - window):
        X.append(feat_scaled[i:i + window])
        yi = i + window - 1
        y.append(labels[yi] if 0 <= yi < len(labels) else 0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

def _cv_macro_f1_score(X: np.ndarray, y: np.ndarray, n_classes: int, folds: int = 3, epochs: int = 3, time_guard=None) -> float:
    if (not _HAS_SKLEARN) or (not _HAS_TORCH):
        return np.nan

    uniq, cnts = np.unique(y, return_counts=True)
    if (len(uniq) < 2) or np.min(cnts) < _MIN_SAMPLES_PER_CLASS:
        return np.nan

    Xf = X.reshape(len(X), -1)
    k = int(min(folds, np.max([2, np.min(cnts)])))
    if k < 2:
        return np.nan

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(os.getenv("GLOBAL_SEED", "20240101")))
    f1s = []
    for tr_idx, va_idx in skf.split(Xf, y):
        if time_guard and (time.time() - time_guard["t0"] > time_guard["budget"]):
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

def _heuristic_score(feat_scaled: np.ndarray, labels: np.ndarray, window: int) -> float:
    if len(feat_scaled) < window or len(labels) < window:
        return -np.inf
    recent = np.nan_to_num(feat_scaled[-window:], nan=0.0, posinf=0.0, neginf=0.0)
    recent_vol = float(np.std(recent, dtype=np.float32))
    diffs = np.diff(labels[-window:])
    label_change = float(np.mean(diffs != 0)) if len(diffs) > 0 else 0.0
    score = recent_vol * (1.0 + label_change)
    return -np.inf if (np.isnan(score) or np.isinf(score)) else score

def _recent_variance(feat_scaled: np.ndarray, window: int) -> float:
    if len(feat_scaled) < window:
        return np.inf
    recent = np.nan_to_num(feat_scaled[-window:], nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.var(recent, dtype=np.float32))

def _apply_variance_penalty(score: float, var: float) -> float:
    if not np.isfinite(score):
        return -np.inf
    if _VAR_PENALTY <= 0:
        return score
    return float(score - _VAR_PENALTY * np.log1p(max(var, 0.0)))

def _normalize_features_df(feat: pd.DataFrame) -> np.ndarray:
    features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
    features_only = features_only.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if features_only.shape[1] < FEATURE_INPUT_SIZE:
        pad = FEATURE_INPUT_SIZE - features_only.shape[1]
        for i in range(pad):
            features_only[f"pad_{i}"] = 0.0
    elif features_only.shape[1] > FEATURE_INPUT_SIZE:
        features_only = features_only.iloc[:, :FEATURE_INPUT_SIZE]
    return features_only.to_numpy(dtype=np.float32)

def find_best_window(symbol: str, strategy: str, window_list=None, group_id=None):
    t0 = time.time()
    if not window_list:
        window_list = [20, 40, 60]

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

    gains = _future_returns_by_timestamp(df, _strategy_horizon_hours(strategy))
    labels = _label_from_future_returns(gains, symbol, strategy, group_id=group_id)
    if labels.size == 0:
        print(f"[find_best_window] 라벨 없음 → fallback={min(window_list)}")
        return int(min(window_list))
    n_classes = int(np.max(labels)) + 1

    best_w, best_score, best_var = int(min(window_list)), -np.inf, np.inf
    for w in sorted(set(int(x) for x in window_list)):
        if len(feat_scaled) < w + 5:
            continue
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

        better = (adj > best_score + 1e-6) or (abs(adj - best_score) <= 1e-6 and var < best_var)
        if better:
            best_score, best_w, best_var = adj, w, var

    if best_score == -np.inf:
        print(f"[find_best_window] 유효 윈도우 없음 → fallback={min(window_list)}")
        return int(min(window_list))

    print(f"[find_best_window] {symbol}-{strategy} -> best={best_w} (score={best_score:.6f}, var={best_var:.6f})")
    return int(best_w)

def find_best_windows(symbol: str, strategy: str, window_list=None, group_id=None):
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

    scored.sort(key=lambda x: (-x[1], x[2]))
    top = [int(w) for w, _, _ in scored[:3]]
    print(f"[find_best_windows] {symbol}-{strategy} -> {top}")
    return top

__all__ = ["find_best_window", "find_best_windows"]
