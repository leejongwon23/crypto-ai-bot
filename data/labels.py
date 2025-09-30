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

    # --- 마지막 구간 우측 포함
    highs_adj = highs.copy()
    highs_adj[-1] = highs[-1] + 1e-12

    # --- 벡터화 binning (우측 포함 처리)
    edges = np.concatenate(([lows[0]], highs_adj), axis=0)  # 길이 C+1
    bins = np.searchsorted(edges, gains, side="right") - 1
    bins = np.clip(bins, 0, len(class_ranges) - 1).astype(np.int64)

    # --- 경계 마스크 1차 적용: ±BOUNDARY_BAND
    gcol = gains.reshape(-1, 1)  # (N,1)
    near_lo = np.abs(gcol - lows.reshape(1, -1)) <= BOUNDARY_BAND
    near_hi = np.abs(gcol - highs.reshape(1, -1)) <= BOUNDARY_BAND
    is_mask = np.any(near_lo | near_hi, axis=1)

    labels[:] = bins
    labels[is_mask] = -1

    # -----------------------------
    # ✅ 가드 1: 마스크 과다 시 자동 완화
    #  - 경계밴드가 너무 넓어 다수 샘플이 -1 되면 밴드를 절반으로 줄여 재적용
    #  - 그래도 과다하면 원상복구(1차 결과 유지)하되 경고 로그만 남김
    # -----------------------------
    try:
        masked_ratio = float((labels == -1).sum()) / float(max(1, n))
    except Exception:
        masked_ratio = 0.0

    if masked_ratio > 0.60 and n > 0:
        try:
            # 밴드 축소 후 재마스킹
            shrink_band = float(BOUNDARY_BAND) * 0.5
            near_lo2 = np.abs(gcol - lows.reshape(1, -1)) <= shrink_band
            near_hi2 = np.abs(gcol - highs.reshape(1, -1)) <= shrink_band
            is_mask2 = np.any(near_lo2 | near_hi2, axis=1)
            labels2 = bins.copy()
            labels2[is_mask2] = -1

            masked_ratio2 = float((labels2 == -1).sum()) / float(max(1, n))
            # 더 나아졌으면 교체
            if masked_ratio2 + 1e-6 < masked_ratio:
                labels = labels2
                logger.info(
                    "make_labels: boundary band shrunk %.4f->%.4f (mask %.2f%% -> %.2f%%) %s/%s",
                    float(BOUNDARY_BAND),
                    shrink_band,
                    masked_ratio * 100.0,
                    masked_ratio2 * 100.0,
                    symbol,
                    strategy,
                )
        except Exception as e:
            logger.warning("make_labels: band shrink safeguard failed: %s", e)

    # -----------------------------
    # ✅ 가드 2: 유효 클래스 수 커버리지 체크
    #  - 마스킹 제외 후 클래스가 1개 이하이면 상위단에서 증강/폴드축소를 하도록
    #    전부 -1로 리턴하여 '학습 보류' 시그널을 보낸다.
    # -----------------------------
    try:
        valid = labels[labels >= 0]
        n_valid = int(valid.size)
        n_unique = int(np.unique(valid).size) if n_valid > 0 else 0
        if n_unique <= 1:
            labels[:] = -1
            logger.warning(
                "make_labels: insufficient class coverage after masking (%d unique). "
                "Mark all as -1 to trigger augmentation/CV fallback. %s/%s",
                n_unique,
                symbol,
                strategy,
            )
    except Exception as e:
        logger.warning("make_labels: coverage check failed: %s", e)

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
