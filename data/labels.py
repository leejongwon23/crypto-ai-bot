# data/labels.py
# 라벨 단일화 모듈 (KST 고정, 창 경계 '< t1')
# - 미래 수익률(gain) 계산 (극단 수익률 방식, config와 동일)
# - config.get_class_ranges() 경계에 따라 클래스 할당
# - (gains, labels, class_ranges) 반환

from __future__ import annotations

import numpy as np
import pandas as pd
from config import (
    get_class_ranges,
    BOUNDARY_BAND,
    _strategy_horizon_hours,
    _future_extreme_signed_returns,
)


def _to_series_ts_kst(ts_like) -> pd.Series:
    """timestamp -> Asia/Seoul timezone-aware Series."""
    ts = pd.to_datetime(ts_like, errors="coerce")
    # KST 고정 (config와 동일)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    return ts


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """
    각 시점 t에서 horizon H 동안의 극단 수익률 기반 '단일' signed return 벡터(길이 N)를 생성.
    - config._future_extreme_signed_returns()는 [dn(<=0), up(>=0)]를 '연결'한 2N 배열을 내보낸다.
      → 여기서는 길이 N으로 복원하기 위해 각 시점에 대해 |dn| vs |up| 중 절댓값이 큰 쪽을 택한다.
    - 타임존: KST(Asia/Seoul) 고정
    - 창 경계: '< t1'
    """
    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)

    horizon_hours = _strategy_horizon_hours(strategy)
    both = _future_extreme_signed_returns(df, horizon_hours=horizon_hours)
    # both = concat([dn, up]) 형태 → N으로 복원
    n = len(df)
    if both.size < 2 * n:  # 가드
        return np.zeros(n, dtype=np.float32)
    dn = both[:n]   # <= 0
    up = both[n:]   # >= 0

    # 각 시점: 더 "큰 절대값"을 갖는 방향을 선택
    abs_dn = np.abs(dn)
    abs_up = np.abs(up)
    choose_up = abs_up >= abs_dn
    gains = np.where(choose_up, up, dn).astype(np.float32)
    return gains


def _assign_label_one(g: float, class_ranges: list[tuple[float, float]]) -> int:
    """
    단일 gain에 대해 구간 인덱스를 반환.
    - 모든 구간은 [lo, hi) 좌폐우개, 단 마지막 구간만 [lo, hi] 포함.
    - 어떤 이유로 모든 구간 밖이라면 가장 가까운 변으로 클립.
    """
    n = len(class_ranges)
    if n == 0:
        return 0
    # 마지막 전까지 [lo, hi)
    for k, (lo, hi) in enumerate(class_ranges[:-1]):
        if (g >= lo) and (g < hi):
            return k
    # 마지막 구간만 [lo, hi]
    lo, hi = class_ranges[-1]
    if (g >= lo) and (g <= hi):
        return n - 1
    # 혹시 바깥이면 클립
    return (0 if g < class_ranges[0][0] else n - 1)


def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    라벨 단일화 엔드포인트.
    1) 미래 수익률 gains 계산(KST, '< t1', 극단 수익률)  → 길이 N
    2) config.get_class_ranges(...)로 경계 획득(고정간격+희소병합/제로밴드 보정)
    3) gains를 경계에 따라 정수 라벨로 매핑
    4) 경계 ±BOUNDARY_BAND 이내 샘플은 -1로 마스킹
    """
    gains = signed_future_return(df, strategy)  # (N,)
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)

    n = gains.shape[0]
    labels = np.empty(n, dtype=np.int64)

    if not class_ranges:
        # 경계가 없으면 전부 -1 처리
        labels.fill(-1)
        return gains.astype(np.float32), labels, class_ranges

    # 경계 배열을 한 번 정리
    ranges_arr = np.array(class_ranges, dtype=float)  # shape (C, 2)
    lows = ranges_arr[:, 0]
    highs = ranges_arr[:, 1]

    # 빠른 마스킹: 어떤 경계와도 BOUNDARY_BAND 이내면 -1
    # (lo, hi 모두에 대해 거리 체크)
    # broadcasting으로 계산
    gcol = gains.reshape(-1, 1)
    near_lo = np.abs(gcol - lows.reshape(1, -1)) <= BOUNDARY_BAND
    near_hi = np.abs(gcol - highs.reshape(1, -1)) <= BOUNDARY_BAND
    is_mask = np.any(near_lo | near_hi, axis=1)

    # 레이블 할당: [lo, hi) except last bin includes hi
    # 먼저 마지막 bin의 hi를 약간 올려 inclusive를 구현(수치적 안정)
    highs_adj = highs.copy()
    highs_adj[-1] = highs[-1] + 1e-12

    # np.searchsorted로 좌폐우개 구현
    # edges = [lo0, hi0, hi1, ..., hi_last(adj)]
    edges = np.concatenate(([lows[0]], highs_adj), axis=0)  # 길이 C+1
    # gains가 어떤 구간에 속하는지: bin = searchsorted(edges, g, side="right") - 1
    bins = np.searchsorted(edges, gains, side="right") - 1
    # 범위 밖은 클립
    bins = np.clip(bins, 0, len(class_ranges) - 1).astype(np.int64)

    # 마스킹 적용
    labels[:] = bins
    labels[is_mask] = -1

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges
