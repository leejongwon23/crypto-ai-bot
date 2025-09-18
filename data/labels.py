# data/labels.py
# 라벨 단일화 모듈 (KST 고정, 창 경계 '< t1')
# - 미래 수익률(gain) 계산
# - config.get_class_ranges() 경계에 따라 클래스 할당
# - (gains, labels, class_ranges) 반환

from __future__ import annotations

import numpy as np
import pandas as pd
from config import get_class_ranges

# 전략별 예측 지평(시간) — config와 동일
_HOURS = {"단기": 4, "중기": 24, "장기": 168}


def _to_series_ts_kst(ts_like) -> pd.Series:
    """timestamp -> Asia/Seoul timezone-aware Series."""
    ts = pd.to_datetime(ts_like, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        # 들어오는 데이터가 naive면 KST로 로컬라이즈
        ts = ts.dt.tz_localize("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    return ts


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """
    각 시점 t에서 horizon H 동안의 종가 기반 미래 수익률.
    - 타임존: KST(Asia/Seoul) 고정
    - 창 경계: '< t1' (t1 직전까지의 마지막 캔들)  → 미래 누수 방지
    - close는 float 강제 + NaN 보간
    """
    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)

    ts = _to_series_ts_kst(df["timestamp"])
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().astype(float).values

    H = pd.Timedelta(hours=_HOURS.get(strategy, 24))
    n = len(df)

    out = np.zeros(n, dtype=np.float32)
    j = 0
    for i in range(n):
        t1 = ts.iloc[i] + H

        # i 이상에서 t1 직전(< t1)까지 포인터 전진
        j = max(j, i)
        while j < n and ts.iloc[j] < t1:
            j += 1

        tgt_idx = max(i, min(j - 1, n - 1))  # 마지막 '< t1'을 사용
        ref = close[i] if close[i] != 0 else (close[i] + 1e-6)
        tgt = close[tgt_idx]
        out[i] = float((tgt - ref) / (ref + 1e-12))

    return out


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
    1) 미래 수익률 gains 계산(KST, '< t1')
    2) config.get_class_ranges(...)로 경계 획득(고정간격+희소병합/제로밴드 보정)
    3) gains를 경계에 따라 정수 라벨로 매핑
    """
    gains = signed_future_return(df, strategy)
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)

    labels = np.zeros(len(gains), dtype=np.int64)
    for i, g in enumerate(gains):
        labels[i] = _assign_label_one(float(g), class_ranges)

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges
