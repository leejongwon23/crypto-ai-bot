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
    get_class_ranges,  # 핵심: 경계는 config에서만 가져옴
)

logger = logging.getLogger(__name__)

# 라벨 안정화 상수
_MIN_CLASS_FRAC = 0.01
_MIN_CLASS_ABS = 8
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
def _strategy_from_hours(hours: int) -> str:
    h = int(max(1, hours))
    if h <= 4:
        return "단기"
    if h <= 24:
        return "중기"
    return "장기"


# -----------------------------
# Target construction
# -----------------------------
def signed_future_return_by_hours(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    """
    미래 구간의 최댓상승/최댓하락 중 절댓값이 큰 쪽의 수익률을 선택하여 부호 포함 반환.
    길이가 N인 입력에 대해 (N,) float32 반환.
    """
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
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if gains.size > 0 and np.all(gains == gains[0]):
        idx = np.arange(n, dtype=np.float32)
        gains = gains + (idx - idx.mean()) * 1e-8
    return gains


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    horizon_hours = _strategy_horizon_hours(strategy)
    return signed_future_return_by_hours(df, horizon_hours=horizon_hours)


# -----------------------------
# Helpers
# -----------------------------
def _ranges_to_edges(ranges: List[Tuple[float, float]]) -> np.ndarray:
    """
    [(lo,hi)] -> 단조 증가 edges[C+1].
    hi가 같거나 역전될 경우 최소 증분을 부여하여 단조성 보장.
    """
    if not ranges:
        return np.array([-1e-6, 0.0, 1e-6], dtype=float)
    e = [float(ranges[0][0])]
    for _, hi in ranges:
        hi_f = float(hi)
        if hi_f <= e[-1]:
            hi_f = e[-1] + 1e-9
        e.append(hi_f)
    for i in range(1, len(e)):
        if e[i] <= e[i - 1]:
            e[i] = e[i - 1] + 1e-9
    return np.asarray(e, dtype=float)


def _vector_bin(gains: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    마지막 엣지만 우측 포함되도록 처리. 결과는 [0, C-1] 범위.
    """
    e = edges.astype(float).copy()
    e[-1] = e[-1] + _EDGE_EPS
    bins = np.searchsorted(e, gains, side="right") - 1
    return np.clip(bins, 0, edges.size - 2).astype(np.int64)


def _coverage(x: np.ndarray) -> int:
    v = x[x >= 0]
    return int(np.unique(v).size) if v.size > 0 else 0


def _masked_ratio(x: np.ndarray) -> float:
    return float((x == -1).sum()) / float(max(1, x.size))


def _require_min_counts(labels: np.ndarray, k: int) -> bool:
    """
    각 클래스 최소 샘플 보장. 부족 시 True를 반환하여 마스크 해제 트리거.
    """
    n = int(labels.size)
    if n == 0:
        return False
    req = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * n)))
    v = labels[labels >= 0]
    if v.size == 0:
        return True
    _, cnts = np.unique(v, return_counts=True)
    return bool((cnts < req).any())


# -----------------------------
# Core binning with boundary mask
# -----------------------------
def _bin_with_boundary_mask(
    gains: np.ndarray,
    edges: np.ndarray,
    symbol: str,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    config에서 받은 edges로 라벨링하고 경계 근처는 -1로 마스킹.
    마스킹 과다 시 단계적으로 완화. 커버리지/최소샘플 보장.
    """
    n = gains.shape[0]
    if edges is None or edges.size < 2:
        logger.warning("labels: empty edges for %s/%s -> safe fallback", symbol, strategy)
        safe_edges = np.array([-1e-6, 0.0, 1e-6], dtype=float)
        return np.full(n, -1, dtype=np.int64), safe_edges

    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

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
    if _masked_ratio(labels) > 0.60:
        cand = _apply_mask(float(BOUNDARY_BAND) * 0.5)
        if _masked_ratio(cand) < _masked_ratio(labels):
            labels = cand
            logger.info(
                "labels: mask ratio reduced %.4f → %.4f (%s/%s)",
                float(BOUNDARY_BAND), float(BOUNDARY_BAND) * 0.5, symbol, strategy
            )

    if _masked_ratio(labels) > 0.60:
        cand = _apply_mask(float(BOUNDARY_BAND) * 0.25)
        if _masked_ratio(cand) < _masked_ratio(labels):
            labels = cand
            logger.info(
                "labels: mask ratio reduced %.4f → %.4f (%s/%s)",
                float(BOUNDARY_BAND), float(BOUNDARY_BAND) * 0.25, symbol, strategy
            )

    # 4) 커버리지 가드. 최악이면 마스킹 해제
    uniq = _coverage(labels)
    if uniq <= 1:
        labels = bins.copy()
        uniq = _coverage(labels)
        logger.warning(
            "labels: boundary mask disabled to recover coverage (uniq=%d) %s/%s",
            uniq, symbol, strategy
        )

    # 5) 최소 샘플 가드. 미달 시 마스크 해제
    k = edges.size - 1
    if _require_min_counts(labels, k):
        labels = bins.copy()
        logger.info("labels: disabled mask due to min-count guard (%s/%s)", symbol, strategy)

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

    # 경계는 config에서만 결정
    ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    if not ranges:
        logger.warning("make_labels: empty class_ranges, using fallback %s/%s", symbol, strategy)
        ranges = [(-1e-6, 0.0), (0.0, 1e-6)]

    edges = _ranges_to_edges(ranges)
    labels, edges_final = _bin_with_boundary_mask(gains, edges, symbol, strategy)

    class_ranges = [(float(edges_final[i]), float(edges_final[i + 1])) for i in range(edges_final.size - 1)]
    edges_count = edges_final.copy()
    edges_count[-1] += _EDGE_EPS
    g_clip = np.clip(gains, edges_final[0], edges_final[-1])
    bin_counts, _ = np.histogram(g_clip, bins=edges_count)
    bin_spans = np.diff(edges_final) * 100.0

    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        edges_final,
        bin_counts.astype(int),
        bin_spans.astype(float),
    )


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

    ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    if not ranges:
        logger.warning("make_labels_for_horizon: empty class_ranges; safe fallback %s/%s", symbol, strategy)
        ranges = [(-1e-6, 0.0), (0.0, 1e-6)]

    edges = _ranges_to_edges(ranges)
    labels, edges_final = _bin_with_boundary_mask(gains, edges, symbol, strategy)

    class_ranges = [(float(edges_final[i]), float(edges_final[i + 1])) for i in range(edges_final.size - 1)]
    edges_count = edges_final.copy()
    edges_count[-1] += _EDGE_EPS
    g_clip = np.clip(gains, edges_final[0], edges_final[-1])
    bin_counts, _ = np.histogram(g_clip, bins=edges_count)
    bin_spans = np.diff(edges_final) * 100.0

    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        strategy,
        edges_final,
        bin_counts.astype(int),
        bin_spans.astype(float),
    )


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
        gains, labels, ranges, strategy, edges, counts, spans = make_labels_for_horizon(
            df, symbol, h, group_id=group_id
        )
        key = f"{h}h" if h < 24 else ("1d" if h == 24 else (f"{h//24}d" if h < 168 else "7d"))
        out[key] = (gains, labels, ranges, edges, counts, spans)
        logger.debug("make_all_horizon_labels: %s -> strategy=%s, key=%s", h, strategy, key)
    return out
