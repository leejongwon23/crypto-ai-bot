from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    BOUNDARY_BAND,
    _strategy_horizon_hours,
    _future_extreme_signed_returns,
)

logger = logging.getLogger(__name__)

# ===== 새 파라미터(환경변수 또는 기본값) =====
_TARGET_BINS = int(os.getenv("TARGET_BINS", "8"))
_OUT_Q_LOW = float(os.getenv("OUTLIER_Q_LOW", "0.01"))   # 1%
_OUT_Q_HIGH = float(os.getenv("OUTLIER_Q_HIGH", "0.99")) # 99%
_MAX_BIN_SPAN_PCT = float(os.getenv("MAX_BIN_SPAN_PCT", "8.0"))  # 단일 bin 폭 상한(절대 %)
_MIN_BIN_COUNT_FRAC = float(os.getenv("MIN_BIN_COUNT_FRAC", "0.05"))  # 최소 샘플 비율

# 라벨 안정화 상수(로컬)
_MIN_CLASS_FRAC = 0.01
_MIN_CLASS_ABS = 8
_Q_EPS = 1e-9
_EDGE_EPS = 1e-12


# -----------------------------
# Timezone helper (KST unified)
# -----------------------------
def _to_series_ts_kst(ts_like) -> pd.Series:
    ts = pd.to_datetime(ts_like, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    return ts


# -----------------------------
# Strategy/Horizon helpers
# -----------------------------
_HOURS2STRATEGY = [(4, "단기"), (24, "중기"), (168, "장기")]

def _strategy_from_hours(hours: int) -> str:
    h = int(max(1, hours))
    if h <= 4: return "단기"
    if h <= 24: return "중기"
    return "장기"


# -----------------------------
# Target construction
# -----------------------------
def signed_future_return_by_hours(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    if (
        df is None
        or len(df) == 0
        or "timestamp" not in df.columns
        or "close" not in df.columns
    ):
        return np.zeros(0, dtype=np.float32)

    both = _future_extreme_signed_returns(df, horizon_hours=int(horizon_hours))
    n = len(df)
    if both is None or both.size < 2 * n:
        return np.zeros(n, dtype=np.float32)

    dn = both[:n]   # <= 0
    up = both[n:]   # >= 0
    gains = np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)
    if np.all(gains == gains[0]):
        idx = np.arange(n, dtype=np.float32)
        gains = gains + (idx - idx.mean()) * 1e-8
    return gains

def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    horizon_hours = _strategy_horizon_hours(strategy)
    return signed_future_return_by_hours(df, horizon_hours=horizon_hours)


# -----------------------------
# Helpers
# -----------------------------
def _vector_bin(gains: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """마지막 엣지만 우측 포함되도록 처리."""
    e = edges.astype(float).copy()
    e[-1] = e[-1] + _EDGE_EPS
    bins = np.searchsorted(e, gains, side="right") - 1
    return np.clip(bins, 0, edges.size - 2).astype(np.int64)

def _coverage(x: np.ndarray) -> int:
    v = x[x >= 0]
    return int(np.unique(v).size) if v.size > 0 else 0

def _needs_rebin(labels: np.ndarray, k: int, n: int) -> bool:
    if n == 0: return False
    req = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * n)))
    vals, cnts = np.unique(labels[labels >= 0], return_counts=True) if (labels >= 0).any() else (np.array([]), np.array([]))
    if vals.size <= 2: return True
    return bool((cnts < req).any())

def _clip_outliers(g: np.ndarray) -> Tuple[np.ndarray, float, float]:
    low = np.quantile(g, _OUT_Q_LOW)
    high = np.quantile(g, _OUT_Q_HIGH)
    if not np.isfinite(low): low = np.min(g)
    if not np.isfinite(high): high = np.max(g)
    return np.clip(g, low, high), float(low), float(high)

def _dedupe_edges(edges: np.ndarray) -> np.ndarray:
    e = edges.astype(float).copy()
    for i in range(1, e.size):
        if not np.isfinite(e[i]): e[i] = e[i-1] + _Q_EPS
        if e[i] <= e[i-1]: e[i] = e[i-1] + _Q_EPS
    return e

def _equal_freq_edges(g: np.ndarray, k: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, k + 1)
    cuts = np.quantile(g, qs)
    return _dedupe_edges(cuts)

def _split_wide_bins(edges: np.ndarray, max_span_pct: float) -> np.ndarray:
    """각 bin 절대폭(% 기준)이 상한 초과 시 내부 균등분할."""
    max_span = max_span_pct / 100.0
    e = edges.tolist()
    i = 0
    while i < len(e) - 1:
        lo, hi = float(e[i]), float(e[i+1])
        span = abs(hi - lo)
        if span > max_span:
            m = int(np.ceil(span / max_span))
            sub = np.linspace(lo, hi, m + 1).tolist()
            e = e[:i] + sub + e[i+2:]
            i += m  # 새로 만든 마지막 구간으로 이동
        else:
            i += 1
    return _dedupe_edges(np.array(e, dtype=float))

def _merge_sparse_bins(edges: np.ndarray, counts: np.ndarray, min_count: int) -> Tuple[np.ndarray, np.ndarray]:
    e = edges.astype(float).tolist()
    c = counts.astype(int).tolist()
    changed = True
    while changed and len(e) > 2:
        changed = False
        for i in range(len(c)):
            if c[i] < min_count:
                # 이웃 중 더 큰 쪽에 병합
                if i == 0:
                    c[i+1] += c[i]; del c[i]; del e[i+1]; changed = True; break
                elif i == len(c) - 1:
                    c[i-1] += c[i]; del c[i]; del e[i]; changed = True; break
                else:
                    if c[i-1] >= c[i+1]:
                        c[i-1] += c[i]; del c[i]; del e[i]; changed = True; break
                    else:
                        c[i+1] += c[i]; del c[i]; del e[i+1]; changed = True; break
    return np.array(e, dtype=float), np.array(c, dtype=int)

def _build_bins(gains: np.ndarray, target_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """edges, counts, spans_pct 생성."""
    x = np.asarray(gains, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        edges = np.array([-0.05, -0.02, -0.005, 0.005, 0.02, 0.05], dtype=float)
        counts = np.zeros(edges.size - 1, dtype=int)
        spans = np.diff(edges) * 100.0
        return edges, counts, spans

    x_clip, lo_q, hi_q = _clip_outliers(x)
    k = max(2, int(target_bins))
    edges = _equal_freq_edges(x_clip, k)

    # 과폭 bin 분할
    edges = _split_wide_bins(edges, _MAX_BIN_SPAN_PCT)

    # 카운트 계산
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    counts, _ = np.histogram(x_clip, bins=edges_count)

    # 희소 bin 병합
    min_count = max(1, int(np.ceil(_MIN_BIN_COUNT_FRAC * x_clip.size)))
    edges, counts = _merge_sparse_bins(edges, counts, min_count)

    spans_pct = np.diff(edges) * 100.0
    return edges.astype(float), counts.astype(int), spans_pct.astype(float)


# -----------------------------
# Core binning with boundary mask + ADAPTIVE REBIN
# -----------------------------
def _bin_with_boundary_mask(
    gains: np.ndarray,
    edges: np.ndarray,
    symbol: str,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = gains.shape[0]
    if edges is None or edges.size < 2:
        logger.warning("labels: empty edges for %s/%s -> safe fallback", symbol, strategy)
        safe_edges = np.array([-1e-6, 0.0, 1e-6], dtype=float)
        return np.full(n, -1, dtype=np.int64), safe_edges

    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)

    # 1) 최초 라벨링
    bins = _vector_bin(gains, edges)

    # 2) 경계 마스킹
    gcol = gains.reshape(-1, 1)
    lows = edges[:-1].reshape(1, -1)
    highs = edges[1:].reshape(1, -1)

    def _apply_mask(eps: float) -> np.ndarray:
        near_lo = np.abs(gcol - lows) <= eps
        near_hi = np.abs(gcol - highs) <= eps
        is_mask = np.any(near_lo | near_hi, axis=1)
        out = bins.copy()
        out[is_mask] = -1
        return out

    labels = _apply_mask(float(BOUNDARY_BAND))

    # 3) 마스킹 과다시 단계 축소
    def _masked_ratio(x: np.ndarray) -> float:
        return float((x == -1).sum()) / float(max(1, x.size))

    if _masked_ratio(labels) > 0.60:
        cand = _apply_mask(float(BOUNDARY_BAND) * 0.5)
        if _masked_ratio(cand) < _masked_ratio(labels):
            labels = cand
            logger.info("labels: mask ratio reduced to %.4f (%s/%s)", float(BOUNDARY_BAND) * 0.5, symbol, strategy)

    if _masked_ratio(labels) > 0.60:
        cand = _apply_mask(float(BOUNDARY_BAND) * 0.25)
        if _masked_ratio(cand) < _masked_ratio(labels):
            labels = cand
            logger.info("labels: mask ratio reduced to %.4f (%s/%s)", float(BOUNDARY_BAND) * 0.25, symbol, strategy)

    # 4) 커버리지 점검. 최악이면 마스킹 해제
    uniq = _coverage(labels)
    if uniq <= 1:
        labels = bins.copy()
        uniq = _coverage(labels)
        logger.warning("labels: boundary mask disabled to recover coverage (uniq=%d) %s/%s", uniq, symbol, strategy)

    # 5) ✅ ADAPTIVE REBIN
    k = edges.size - 1
    n = gains.size
    req_min = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * max(1, n))))
    trigger = (uniq <= 2) or _needs_rebin(labels, k, n)
    if trigger and k >= 3 and n > 0:
        try:
            edges2, _, _ = _build_bins(gains, k)
            labels_dyn = _vector_bin(gains, edges2)
            uniq_dyn = _coverage(labels_dyn)
            vals, cnts = np.unique(labels_dyn, return_counts=True)
            ok_min = (cnts[cnts > 0].min() >= req_min) if cnts.size > 0 else False
            if (uniq_dyn > uniq) or ok_min:
                logger.info("labels: ADAPTIVE_REBIN applied (uniq %d→%d, min_req=%d) %s/%s",
                            uniq, uniq_dyn, req_min, symbol, strategy)
                return labels_dyn.astype(np.int64), edges2.astype(float)
        except Exception as e:
            logger.warning("labels: adaptive rebin failed: %s", e)

    return labels.astype(np.int64), edges.astype(float)


# -----------------------------
# Public API (strategy-based)
# -----------------------------
def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        gains:       float32 (N,)
        labels:      int64   (N,)  (-1 or 0..C-1)
        class_ranges:List[(lo, hi)]
        bin_edges:   float64 (C+1,)
        bin_counts:  int64   (C,)
        bin_spans:   float64 (C,)  # 절대 %
    """
    gains = signed_future_return(df, strategy)  # (N,)
    edges, _, _ = _build_bins(gains, _TARGET_BINS)
    labels, edges_final = _bin_with_boundary_mask(gains, edges, symbol, strategy)

    class_ranges = [(float(edges_final[i]), float(edges_final[i+1])) for i in range(edges_final.size - 1)]
    edges_count = edges_final.copy(); edges_count[-1] += _EDGE_EPS
    bin_counts, _ = np.histogram(np.clip(gains, edges_final[0], edges_final[-1]), bins=edges_count)
    bin_spans = np.diff(edges_final) * 100.0

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges, edges_final, bin_counts.astype(int), bin_spans.astype(float)


# -----------------------------
# Public API (explicit horizons)
# -----------------------------
def make_labels_for_horizon(
    df: pd.DataFrame,
    symbol: str,
    horizon_hours: int,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        gains, labels, class_ranges, mapped_strategy, bin_edges, bin_counts, bin_spans
    """
    gains = signed_future_return_by_hours(df, horizon_hours=int(horizon_hours))
    strategy = _strategy_from_hours(int(horizon_hours))

    edges, _, _ = _build_bins(gains, _TARGET_BINS)
    labels, edges_final = _bin_with_boundary_mask(gains, edges, symbol, strategy)

    class_ranges = [(float(edges_final[i]), float(edges_final[i+1])) for i in range(edges_final.size - 1)]
    edges_count = edges_final.copy(); edges_count[-1] += _EDGE_EPS
    bin_counts, _ = np.histogram(np.clip(gains, edges_final[0], edges_final[-1]), bins=edges_count)
    bin_spans = np.diff(edges_final) * 100.0

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges, strategy, edges_final, bin_counts.astype(int), bin_spans.astype(float)


def make_all_horizon_labels(
    df: pd.DataFrame,
    symbol: str,
    horizons: List[int] | None = None,
    group_id: int | None = None,
) -> Dict[str, tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]]:
    """
    기본: [4, 24, 168]
    키: "4h", "1d", "7d"
    값: (gains, labels, class_ranges, bin_edges, bin_counts, bin_spans)
    """
    if horizons is None:
        horizons = [4, 24, 168]

    out: Dict[str, tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]] = {}
    for h in horizons:
        gains, labels, ranges, strategy, edges, counts, spans = make_labels_for_horizon(df, symbol, h, group_id=group_id)
        key = f"{h}h" if h < 24 else ("1d" if h == 24 else (f"{h//24}d" if h < 168 else "7d"))
        out[key] = (gains, labels, ranges, edges, counts, spans)
        logger.debug("make_all_horizon_labels: %s -> strategy=%s, key=%s", h, strategy, key)
    return out
