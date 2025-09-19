# data/labels.py
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    get_class_ranges,
    BOUNDARY_BAND,
    _strategy_horizon_hours,
    _future_extreme_signed_returns,
)

logger = logging.getLogger(__name__)


# -----------------------------
# Timezone helper (KST unified)
# -----------------------------
def _to_series_ts_kst(ts_like) -> pd.Series:
    """timestamp -> Asia/Seoul timezone-aware Series."""
    ts = pd.to_datetime(ts_like, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    return ts


# -----------------------------
# Target construction
# -----------------------------
def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """
    각 시점 t에서 전략별 horizon H 동안의 '극단 수익률'을 단일 signed 값으로 생성.
    - up(>=0)과 dn(<=0) 중 절대값이 큰 쪽을 채택
    - 반환: shape (N,) float32
    """
    if (
        df is None
        or len(df) == 0
        or "timestamp" not in df.columns
        or "close" not in df.columns
    ):
        return np.zeros(0, dtype=np.float32)

    horizon_hours = _strategy_horizon_hours(strategy)
    both = _future_extreme_signed_returns(df, horizon_hours=horizon_hours)

    n = len(df)
    if both is None or both.size < 2 * n:
        # 상위 단계에서 graceful 처리되도록 안전 반환
        return np.zeros(n, dtype=np.float32)

    dn = both[:n]   # <= 0
    up = both[n:]   # >= 0

    # 절대값 큰 쪽 선택
    gains = np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)

    # NaN/inf 방어
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(
        np.float32
    )
    return gains


def _assign_label_one(g: float, class_ranges: List[Tuple[float, float]]) -> int:
    """(디버그/안전용) 스칼라 한 개를 구간에 매핑."""
    n = len(class_ranges)
    if n == 0:
        return 0
    for k, (lo, hi) in enumerate(class_ranges[:-1]):
        if (g >= lo) and (g < hi):
            return k
    lo, hi = class_ranges[-1]
    if (g >= lo) and (g <= hi):
        return n - 1
    return 0 if g < class_ranges[0][0] else n - 1


# -----------------------------
# Labeling (with boundary mask)
# -----------------------------
def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    1) gains 계산
    2) class_ranges 획득 (config.get_class_ranges)
    3) 경계 ±BOUNDARY_BAND 이내는 -1 마스킹
    4) 벡터화 binning (마지막 구간 우측 포함)

    Returns:
        gains:  float32 (N,)
        labels: int64   (N,)  (-1 or 0..C-1)
        class_ranges: List[(lo, hi)]
    """
    gains = signed_future_return(df, strategy)  # (N,)
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)

    n = gains.shape[0]
    labels = np.empty(n, dtype=np.int64)

    if not class_ranges:
        labels.fill(-1)
        logger.warning(
            "make_labels: no class_ranges for %s/%s -> all masked", symbol, strategy
        )
        return gains.astype(np.float32), labels, class_ranges

    # 안전 보정(재확인)
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(
        np.float32
    )

    ranges_arr = np.asarray(class_ranges, dtype=float)  # (C, 2)
    lows = ranges_arr[:, 0]
    highs = ranges_arr[:, 1]

    # --- 경계 마스크: ±BOUNDARY_BAND 이내 샘플 -1
    gcol = gains.reshape(-1, 1)  # (N,1)
    near_lo = np.abs(gcol - lows.reshape(1, -1)) <= BOUNDARY_BAND
    near_hi = np.abs(gcol - highs.reshape(1, -1)) <= BOUNDARY_BAND
    is_mask = np.any(near_lo | near_hi, axis=1)

    # --- 마지막 구간 우측 포함
    highs_adj = highs.copy()
    highs_adj[-1] = highs[-1] + 1e-12

    # --- 벡터화 binning
    edges = np.concatenate(([lows[0]], highs_adj), axis=0)  # 길이 C+1
    bins = np.searchsorted(edges, gains, side="right") - 1
    bins = np.clip(bins, 0, len(class_ranges) - 1).astype(np.int64)

    labels[:] = bins
    labels[is_mask] = -1

    # --- 방어 검사
    if not ((labels == -1) | ((labels >= 0) & (labels < len(class_ranges)))).all():
        # 범위를 벗어나면 마스킹
        labels = np.where(
            (labels < -1) | (labels >= len(class_ranges)), -1, labels
        ).astype(np.int64)
        logger.error(
            "make_labels: label out-of-range detected and masked for %s/%s",
            symbol,
            strategy,
        )

    # --- 분포 요약(디버그)
    try:
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.debug(
            "make_labels: %s/%s labels distribution (incl -1 mask): %s",
            symbol,
            strategy,
            dist,
        )
    except Exception:
        pass

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges
