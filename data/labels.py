# data/labels.py
from __future__ import annotations

import numpy as np
import pandas as pd
from config import (
    get_class_ranges,
    BOUNDARY_BAND,
    _strategy_horizon_hours,
    _future_extreme_signed_returns,
)
import logging

logger = logging.getLogger(__name__)


def _to_series_ts_kst(ts_like) -> pd.Series:
    """timestamp -> Asia/Seoul timezone-aware Series."""
    ts = pd.to_datetime(ts_like, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    return ts


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """
    각 시점 t에서 horizon H 동안의 극단 수익률 기반 '단일' signed return 벡터(길이 N)를 생성.
    """
    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)

    horizon_hours = _strategy_horizon_hours(strategy)
    both = _future_extreme_signed_returns(df, horizon_hours=horizon_hours)
    n = len(df)
    if both is None or both.size < 2 * n:
        # 안전하게 0 리턴 (상위단에서 처리 가능하도록)
        return np.zeros(n, dtype=np.float32)
    dn = both[:n]   # <= 0
    up = both[n:]   # >= 0

    abs_dn = np.abs(dn)
    abs_up = np.abs(up)
    choose_up = abs_up >= abs_dn
    gains = np.where(choose_up, up, dn).astype(np.float32)

    # NaN/inf 보호
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return gains


def _assign_label_one(g: float, class_ranges: list[tuple[float, float]]) -> int:
    n = len(class_ranges)
    if n == 0:
        return 0
    for k, (lo, hi) in enumerate(class_ranges[:-1]):
        if (g >= lo) and (g < hi):
            return k
    lo, hi = class_ranges[-1]
    if (g >= lo) and (g <= hi):
        return n - 1
    return (0 if g < class_ranges[0][0] else n - 1)


def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    1) gains 계산
    2) class_ranges 획득
    3) labels 매핑
    4) 경계 ±BOUNDARY_BAND 이내 샘플은 -1로 마스킹
    반환: (gains(float32), labels(int64), class_ranges)
    """
    gains = signed_future_return(df, strategy)  # (N,)
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)

    n = gains.shape[0]
    labels = np.empty(n, dtype=np.int64)

    if not class_ranges:
        labels.fill(-1)
        logger.warning("make_labels: no class_ranges for %s/%s -> all masked", symbol, strategy)
        return gains.astype(np.float32), labels, class_ranges

    # 안전: gains에 NaN/inf 없애기 (재확인)
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    ranges_arr = np.array(class_ranges, dtype=float)  # shape (C, 2)
    lows = ranges_arr[:, 0]
    highs = ranges_arr[:, 1]

    # 마스크 계산 (경계 근처)
    gcol = gains.reshape(-1, 1)
    near_lo = np.abs(gcol - lows.reshape(1, -1)) <= BOUNDARY_BAND
    near_hi = np.abs(gcol - highs.reshape(1, -1)) <= BOUNDARY_BAND
    is_mask = np.any(near_lo | near_hi, axis=1)

    # last high inclusive 구현
    highs_adj = highs.copy()
    highs_adj[-1] = highs[-1] + 1e-12

    edges = np.concatenate(([lows[0]], highs_adj), axis=0)  # 길이 C+1
    bins = np.searchsorted(edges, gains, side="right") - 1
    bins = np.clip(bins, 0, len(class_ranges) - 1).astype(np.int64)

    labels[:] = bins
    labels[is_mask] = -1

    # 방어 검사: labels는 -1 또는 0..C-1 이어야 함
    valid_mask = ((labels >= -1) & (labels < len(class_ranges))).all()
    if not valid_mask:
        # 안전한 핸들링: 범위 밖은 클립/마스크
        labels = np.where((labels < -1) | (labels >= len(class_ranges)), -1, labels).astype(np.int64)
        logger.error("make_labels: label out-of-range detected and masked for %s/%s", symbol, strategy)

    # 추가 assert (개발용) — 운영에서는 예외로 바꾸기 힘드므로 로그만
    if not ((labels == -1) | ((labels >= 0) & (labels < len(class_ranges)))).all():
        logger.error("make_labels: invariant violated labels range for %s/%s", symbol, strategy)

    # 요약 로그 (작게)
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    logger.debug("make_labels: %s/%s labels distribution (incl -1 mask): %s", symbol, strategy, dist)

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges
