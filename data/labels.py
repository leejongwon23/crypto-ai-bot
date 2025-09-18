# data/labels.py
# 라벨 단일화 모듈
# - 미래 수익률(gain) 계산
# - config.get_class_ranges()로 얻은 경계에 따라 클래스 할당
# - (gains, labels, class_ranges) 반환

from __future__ import annotations

import numpy as np
import pandas as pd

from config import get_class_ranges

# 전략별 예측 지평(시간) — config 내부 구현과 동일 값 사용
_HOURS = {"단기": 4, "중기": 24, "장기": 168}


def _to_series_ts(ts_like) -> pd.Series:
    """timestamp 컬럼을 안전하게 timezone-aware Series로 변환(Asia/Seoul)."""
    ts = pd.to_datetime(ts_like, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        # 들어오는 데이터가 naive면 UTC로 간주 후 KST로 변환
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    return ts


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """
    각 시점 t에서 전략별 horizon H 이후(또는 그 직전 인덱스)의 종가 대비 수익률.
    - 오염 방지: 현재 시점 이후의 정보만 사용.
    - 실패/결측 방어: close를 숫자로 강제 변환, NaN은 앞뒤 보간.
    """
    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)

    ts = _to_series_ts(df["timestamp"])
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().astype(float).values

    H = pd.Timedelta(hours=_HOURS.get(strategy, 24))

    out = np.zeros(len(df), dtype=np.float32)
    j = 0
    for i in range(len(df)):
        t1 = ts.iloc[i] + H

        # i 이후에서 t1 직전까지 전진
        while j < len(df) and ts.iloc[j] < t1:
            j += 1

        ref = close[i]
        tgt_idx = min(j, len(df) - 1)  # t1을 넘었으면 마지막 직전 인덱스 또는 마지막
        tgt = close[tgt_idx]

        # 안정적 비율 계산
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
    # 마지막 구간 포함
    lo, hi = class_ranges[-1]
    if g >= lo:
        return n - 1
    # 아래쪽 언더플로 방어
    return 0


def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """
    라벨 단일화 엔드포인트.
    1) 미래 수익률 gains 계산
    2) config.get_class_ranges(...)로 경계 획득(고정간격 + 희소병합/제로밴드 보정)
    3) gains를 경계에 따라 정수 라벨로 매핑
    Returns
    -------
    gains : np.ndarray(float32)  # len(df)
    labels: np.ndarray(int64)    # len(df)
    class_ranges: list[(lo, hi)] # 사용한 클래스 경계(표시용/후처리용)
    """
    gains = signed_future_return(df, strategy)

    # 그룹 슬라이싱이 필요하면 group_id 넘겨서 일부만 가져올 수 있음(없으면 전체)
    class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)

    # 라벨링
    labels = np.zeros(len(gains), dtype=np.int64)
    for i, g in enumerate(gains):
        labels[i] = _assign_label_one(float(g), class_ranges)

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges
