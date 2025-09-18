# data/labels.py
# 라벨 단일화 모듈 (KST 고정, 창 경계 '< t1')
# - 미래 수익률(gain) 계산 (극단 수익률 방식, config와 동일)
# - config.get_class_ranges() 경계에 따라 클래스 할당
# - (gains, labels, class_ranges) 반환

from __future__ import annotations

import numpy as np
import pandas as pd
from config import get_class_ranges, BOUNDARY_BAND, _strategy_horizon_hours, _future_extreme_signed_returns


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
    각 시점 t에서 horizon H 동안의 극단 수익률(최대 상승/최대 하락) 기반 signed return.
    - config._future_extreme_signed_returns()와 동일 정의 사용
    - 타임존: KST(Asia/Seoul) 고정
    - 창 경계: '< t1'
    """
    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)

    horizon_hours = _strategy_horizon_hours(strategy)
    rets_signed = _future_extreme_signed_returns(df, horizon_hours=horizon_hours)
    return rets_signed


def _assign_label_one(g: float, class_ranges: list[tuple[float, float]]) -> int:
    """
    단일 gain에 대해 구간 인덱스를 반환.
    - 모든 구간은 [lo, hi) 좌폐우개, 단 마지막 구간만 [lo, hi] 포함.
    - 어떤 이유로 모든 구간 밖이라면 가장 가까운 변으로 클립.
    """
    n = len(class_ranges)
    if n == 0:
        return 0
    for k, (lo, hi) in enumerate(class_ranges[:-1]):
        if (g >= lo) and (g < hi):
            return k
    lo, hi = class_ranges[-1]
    return n - 1 if g >= lo else 0


def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    라벨 단일화 엔드포인트.
    1) 미래 수익률 gains 계산(KST, '< t1', 극단 수익률)
    2) config.get_class_ranges(...)로 경계 획득(고정간격+희소병합/제로밴드 보정)
    3) gains를 경계에 따라 정수 라벨로 매핑
    4) 경계 ±BOUNDARY_BAND 이내 샘플은 -1로 마스킹
    """
    gains = signed_future_return(df, strategy)
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)

    labels = np.zeros(len(gains), dtype=np.int64)
    for i, g in enumerate(gains):
        lbl = _assign_label_one(float(g), class_ranges)
        # 경계 근처 샘플은 마스킹 처리
        for lo, hi in class_ranges:
            if abs(g - lo) <= BOUNDARY_BAND or abs(g - hi) <= BOUNDARY_BAND:
                lbl = -1
                break
        labels[i] = lbl

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges
