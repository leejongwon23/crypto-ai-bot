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

# 라벨 안정화 상수(로컬). 외부 설정 없이 보수적으로 동작.
_MIN_CLASS_FRAC = 0.01   # 각 클래스 최소 비율 목표치(1%)
_MIN_CLASS_ABS = 8       # 각 클래스 최소 샘플 수 하한
_Q_EPS = 1e-9            # 분위 경계 보정용
_EDGE_EPS = 1e-12        # 최종 우측 포함 엣지 보정


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

    # NaN/inf 방어 + 극단적 미동장 구간의 완전 동률을 미세 분리
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)
    if np.all(gains == gains[0]):
        # 전구간 동일값이면 미세 잡음 추가(결정경계 분리용). 재현가능성을 위해 작은 등차 시퀀스.
        idx = np.arange(n, dtype=np.float32)
        gains = gains + (idx - idx.mean()) * 1e-8

    return gains


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """기존 호환: 전략명(단/중/장)으로 horizon을 가져와서 계산."""
    horizon_hours = _strategy_horizon_hours(strategy)
    return signed_future_return_by_hours(df, horizon_hours=horizon_hours)


def _assign_label_one(g: float, class_ranges: List[Tuple[float, float]]) -> int:
    """(디버그/안전용) 스칼라 한 개를 구간에 매핑."""
    n = len(class_ranges)
    if n == 0:
        return -1
    for k, (lo, hi) in enumerate(class_ranges[:-1]):
        if (g >= lo) and (g < hi):
            return k
    lo, hi = class_ranges[-1]
    if (g >= lo) and (g <= hi):
        return n - 1
    return 0 if g < class_ranges[0][0] else n - 1


# -----------------------------
# Helpers
# -----------------------------
def _vector_bin(gains: np.ndarray, ranges: List[Tuple[float, float]]) -> np.ndarray:
    """마지막 구간만 우측 포함하여 벡터화 bin."""
    arr = np.asarray(ranges, dtype=float)
    lows, highs = arr[:, 0], arr[:, 1]
    highs_adj = highs.copy()
    highs_adj[-1] = highs[-1] + _EDGE_EPS
    edges = np.concatenate(([lows[0]], highs_adj), axis=0)
    bins = np.searchsorted(edges, gains, side="right") - 1
    return np.clip(bins, 0, len(ranges) - 1).astype(np.int64)

def _quantile_ranges(gains: np.ndarray, k: int) -> List[Tuple[float, float]]:
    """
    데이터 분포로부터 동적으로 k개 구간 경계를 생성(동일카운트 분위).
    마지막 구간만 우측 포함 규칙에 맞게 정렬.
    """
    x = np.asarray(gains, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        # 안전 기본값
        return [(-0.05, -0.02), (-0.02, -0.005), (-0.005, 0.005), (0.005, 0.02), (0.02, 0.05)][:k]

    qs = np.linspace(0.0, 1.0, k + 1)
    cuts = np.quantile(x, qs)

    # 단조 위반/중복 방지 미세 보정
    for i in range(1, cuts.size):
        if not np.isfinite(cuts[i]):
            cuts[i] = cuts[i - 1] + _Q_EPS
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + _Q_EPS

    # 구간 폭이 0에 수렴하면 소폭 벌려서 빈 구간 방지
    for i in range(k):
        if (cuts[i + 1] - cuts[i]) < _Q_EPS:
            mid = (cuts[i + 1] + cuts[i]) * 0.5
            cuts[i] = mid - _Q_EPS * 0.5
            cuts[i + 1] = mid + _Q_EPS * 0.5

    ranges = [(float(cuts[i]), float(cuts[i + 1])) for i in range(k)]
    return ranges

def _coverage(x: np.ndarray) -> int:
    v = x[x >= 0]
    return int(np.unique(v).size) if v.size > 0 else 0

def _needs_rebin(labels: np.ndarray, k: int, n: int) -> bool:
    """클래스 분포가 지나치게 빈약하면 재라벨 필요."""
    if n == 0:
        return False
    req = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * n)))
    vals, cnts = np.unique(labels[labels >= 0], return_counts=True) if (labels >= 0).any() else (np.array([]), np.array([]))
    if vals.size <= 2:
        return True
    return bool((cnts < req).any())


# -----------------------------
# Core binning with boundary mask + ADAPTIVE REBIN
# -----------------------------
def _bin_with_boundary_mask(
    gains: np.ndarray,
    class_ranges: List[Tuple[float, float]],
    symbol: str,
    strategy: str,
) -> np.ndarray:
    """
    - 기본: 주어진 class_ranges로 라벨링(마지막 구간만 우측 포함)
    - 경계 ±BOUNDARY_BAND 마스킹
    - 마스킹 과다(>60%) 시 밴드 1/2→1/4로 축소, 그래도 나쁘면 마스킹 해제
    - 커버리지/최소샘플 미달 시 ✅ ADAPTIVE REBIN(분위 경계)로 재라벨링
    """
    n = gains.shape[0]
    labels = np.empty(n, dtype=np.int64)

    if not class_ranges:
        labels.fill(-1)
        logger.warning("labels: no class_ranges for %s/%s -> all masked", symbol, strategy)
        return labels

    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)

    # 1) 최초 라벨링
    bins = _vector_bin(gains, class_ranges)

    # 2) 경계 마스킹
    arr = np.asarray(class_ranges, dtype=float)
    lows, highs = arr[:, 0], arr[:, 1]
    gcol = gains.reshape(-1, 1)

    def _apply_mask(eps: float) -> np.ndarray:
        near_lo = np.abs(gcol - lows.reshape(1, -1)) <= eps
        near_hi = np.abs(gcol - highs.reshape(1, -1)) <= eps
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
            logger.info("labels: mask ratio reduced by shrinking band to %.4f (%s/%s)", float(BOUNDARY_BAND) * 0.5, symbol, strategy)

    if _masked_ratio(labels) > 0.60:
        cand = _apply_mask(float(BOUNDARY_BAND) * 0.25)
        if _masked_ratio(cand) < _masked_ratio(labels):
            labels = cand
            logger.info("labels: mask ratio reduced by shrinking band to %.4f (%s/%s)", float(BOUNDARY_BAND) * 0.25, symbol, strategy)

    # 4) 커버리지 점검. 최악이면 마스킹 해제
    uniq = _coverage(labels)
    if uniq <= 1:
        labels = bins.copy()
        uniq = _coverage(labels)
        logger.warning("labels: boundary mask disabled to recover coverage (uniq=%d) %s/%s", uniq, symbol, strategy)

    # 5) ✅ ADAPTIVE REBIN 트리거: 유효 클래스<=2 또는 최소샘플 미달
    k = len(class_ranges)
    req_min = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * max(1, n))))
    trigger = (uniq <= 2) or _needs_rebin(labels, k, n)
    if trigger and k >= 3 and n > 0:
        try:
            dyn_ranges = _quantile_ranges(gains, k=k)
            labels_dyn = _vector_bin(gains, dyn_ranges)
            uniq_dyn = _coverage(labels_dyn)

            # 최소샘플 조건 점검
            vals, cnts = np.unique(labels_dyn, return_counts=True)
            ok_min = (cnts[cnts > 0].min() >= req_min) if cnts.size > 0 else False

            if (uniq_dyn > uniq) or ok_min:
                labels = labels_dyn
                logger.info(
                    "labels: ADAPTIVE_REBIN applied (uniq %d→%d, min_req=%d) %s/%s",
                    uniq, uniq_dyn, req_min, symbol, strategy
                )
        except Exception as e:
            logger.warning("labels: adaptive rebin failed: %s", e)

    # 6) 안전 범위 검사
    if not ((labels == -1) | ((labels >= 0) & (labels < k))).all():
        labels = np.where((labels < -1) | (labels >= k), -1, labels).astype(np.int64)
        logger.error("labels: out-of-range detected and masked for %s/%s", symbol, strategy)

    # 7) 분포 로그(디버그)
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
