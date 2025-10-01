# data/labels.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    get_class_ranges,          # (symbol, strategy, group_id) -> List[(lo, hi)]
    BOUNDARY_BAND,            # 하드네거티브 경계 밴드(ε)
    _strategy_horizon_hours,  # "단기/중기/장기" -> 4/24/168
    _future_extreme_signed_returns,  # df, horizon_hours -> concat([dn(<=0), up(>=0)])
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
# Strategy/Horizon helpers
# -----------------------------
_HOURS2STRATEGY = [
    (4, "단기"),
    (24, "중기"),
    (168, "장기"),
]

def _strategy_from_hours(hours: int) -> str:
    """4h/24h/168h를 단기/중기/장기로 매핑(기본: 장기)."""
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
    각 시점 t에서 horizon_hours 동안의 '극단 수익률'을 단일 signed 값으로 생성.
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

    both = _future_extreme_signed_returns(df, horizon_hours=int(horizon_hours))

    n = len(df)
    if both is None or both.size < 2 * n:
        # 상위 단계에서 graceful 처리되도록 안전 반환
        return np.zeros(n, dtype=np.float32)

    dn = both[:n]   # <= 0
    up = both[n:]   # >= 0

    # 절대값 큰 쪽 선택
    gains = np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)

    # NaN/inf 방어
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)
    return gains


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """기존 호환: 전략명(단/중/장)으로 horizon을 가져와서 계산."""
    horizon_hours = _strategy_horizon_hours(strategy)
    return signed_future_return_by_hours(df, horizon_hours=horizon_hours)


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
# Core binning with boundary mask
# -----------------------------
def _bin_with_boundary_mask(
    gains: np.ndarray,
    class_ranges: List[Tuple[float, float]],
    symbol: str,
    strategy: str,
) -> np.ndarray:
    """
    - 마지막 구간만 우측 포함
    - 경계 ±BOUNDARY_BAND 이내는 -1 마스킹
    - 마스킹 과다(>60%) 시 밴드 1/2로 축소하여 1회 재시도
    - 커버리지(유효 클래스 수) <= 1 이면 폴백 수행
    """
    n = gains.shape[0]
    labels = np.empty(n, dtype=np.int64)

    if not class_ranges:
        labels.fill(-1)
        logger.warning("labels: no class_ranges for %s/%s -> all masked", symbol, strategy)
        return labels

    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)

    ranges_arr = np.asarray(class_ranges, dtype=float)  # (C, 2)
    lows = ranges_arr[:, 0]
    highs = ranges_arr[:, 1]

    # 마지막 구간 우측 포함
    highs_adj = highs.copy()
    highs_adj[-1] = highs[-1] + 1e-12

    # 벡터화 binning
    edges = np.concatenate(([lows[0]], highs_adj), axis=0)  # 길이 C+1
    bins = np.searchsorted(edges, gains, side="right") - 1
    bins = np.clip(bins, 0, len(class_ranges) - 1).astype(np.int64)

    # 경계 마스킹 1차: ±ε
    gcol = gains.reshape(-1, 1)
    near_lo = np.abs(gcol - lows.reshape(1, -1)) <= BOUNDARY_BAND
    near_hi = np.abs(gcol - highs.reshape(1, -1)) <= BOUNDARY_BAND
    is_mask = np.any(near_lo | near_hi, axis=1)

    labels[:] = bins
    labels[is_mask] = -1

    # 마스킹 과다 방지: 60% 초과 시 ε 절반으로 축소하여 1회 재적용
    try:
        masked_ratio = float((labels == -1).sum()) / float(max(1, n))
    except Exception:
        masked_ratio = 0.0

    if masked_ratio > 0.60 and n > 0:
        try:
            shrink_band = float(BOUNDARY_BAND) * 0.5
            near_lo2 = np.abs(gcol - lows.reshape(1, -1)) <= shrink_band
            near_hi2 = np.abs(gcol - highs.reshape(1, -1)) <= shrink_band
            is_mask2 = np.any(near_lo2 | near_hi2, axis=1)
            labels2 = bins.copy()
            labels2[is_mask2] = -1

            masked_ratio2 = float((labels2 == -1).sum()) / float(max(1, n))
            if masked_ratio2 + 1e-6 < masked_ratio:
                labels = labels2
                logger.info(
                    "labels: boundary band shrunk %.4f->%.4f (mask %.2f%% -> %.2f%%) %s/%s",
                    float(BOUNDARY_BAND), shrink_band,
                    masked_ratio * 100.0, masked_ratio2 * 100.0,
                    symbol, strategy,
                )
        except Exception as e:
            logger.warning("labels: band shrink safeguard failed: %s", e)

    # 커버리지 가드(강화): 유효 클래스 수 <=1 이면 순차 폴백
    try:
        def _coverage_count(x):
            v = x[x >= 0]
            return (int(v.size), int(np.unique(v).size) if v.size > 0 else 0)

        n_valid, n_unique = _coverage_count(labels)
        if n_unique <= 1 and n > 0:
            # 1) 밴드 1/4로 재시도
            shrink_band = float(BOUNDARY_BAND) * 0.25
            near_lo3 = np.abs(gcol - lows.reshape(1, -1)) <= shrink_band
            near_hi3 = np.abs(gcol - highs.reshape(1, -1)) <= shrink_band
            is_mask3 = np.any(near_lo3 | near_hi3, axis=1)
            labels3 = bins.copy()
            labels3[is_mask3] = -1
            _, u3 = _coverage_count(labels3)
            if u3 > 1:
                labels = labels3
                logger.info(
                    "labels: coverage fix by shrinking band to %.4f (%s/%s)",
                    shrink_band, symbol, strategy
                )
            else:
                # 2) 최후 폴백: 경계 마스킹 해제(빈즈 그대로 사용)
                labels = bins.copy()
                _, u4 = _coverage_count(labels)
                logger.warning(
                    "labels: boundary mask disabled to recover coverage (unique=%d->%d) %s/%s",
                    n_unique, u4, symbol, strategy
                )
    except Exception as e:
        logger.warning("labels: coverage check failed: %s", e)

    # 안전 범위 검사
    if not ((labels == -1) | ((labels >= 0) & (labels < len(class_ranges)))).all():
        labels = np.where((labels < -1) | (labels >= len(class_ranges)), -1, labels).astype(np.int64)
        logger.error("labels: out-of-range detected and masked for %s/%s", symbol, strategy)

    # 분포 로그(디버그)
    try:
        u, c = np.unique(labels, return_counts=True)
        logger.debug("labels: %s/%s distribution (incl -1): %s", symbol, strategy, dict(zip(u.tolist(), c.tolist())))
    except Exception:
        pass

    return labels.astype(np.int64)


# -----------------------------
# Public API (strategy-based)
# -----------------------------
def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    기존 호환 함수: 전략별 horizon으로 gains/labels 생성.
    Returns:
        gains:  float32 (N,)
        labels: int64   (N,)  (-1 or 0..C-1)
        class_ranges: List[(lo, hi)]
    """
    gains = signed_future_return(df, strategy)  # (N,)
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    labels = _bin_with_boundary_mask(gains, class_ranges, symbol, strategy)
    return gains.astype(np.float32), labels.astype(np.int64), class_ranges


# -----------------------------
# Public API (explicit horizons)
# -----------------------------
def make_labels_for_horizon(
    df: pd.DataFrame,
    symbol: str,
    horizon_hours: int,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], str]:
    """
    horizon_hours를 명시적으로 주어 라벨 생성(+전략명 반환).
    - 4h → 단기, 24h → 중기, 168h → 장기 (근처 시간도 가장 가까운 전략으로 매핑)
    Returns:
        gains, labels, class_ranges, mapped_strategy
    """
    gains = signed_future_return_by_hours(df, horizon_hours=int(horizon_hours))
    strategy = _strategy_from_hours(int(horizon_hours))
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
    labels = _bin_with_boundary_mask(gains, class_ranges, symbol, strategy)
    return gains.astype(np.float32), labels.astype(np.int64), class_ranges, strategy


def make_all_horizon_labels(
    df: pd.DataFrame,
    symbol: str,
    horizons: List[int] | None = None,
    group_id: int | None = None,
) -> Dict[str, tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]]:
    """
    여러 horizon(+4h/+1d/+7d)을 한 번에 계산해 dict로 반환.
    기본: [4, 24, 168]
    키: "4h", "1d", "7d"
    """
    if horizons is None:
        horizons = [4, 24, 168]

    out: Dict[str, tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]] = {}
    for h in horizons:
        gains, labels, ranges, strategy = make_labels_for_horizon(df, symbol, h, group_id=group_id)
        key = f"{h}h" if h < 24 else ("1d" if h == 24 else (f"{h//24}d" if h < 168 else "7d"))
        out[key] = (gains, labels, ranges)
        logger.debug("make_all_horizon_labels: %s -> strategy=%s, key=%s", h, strategy, key)
    return out
