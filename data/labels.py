from __future__ import annotations

import json
import logging
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    BOUNDARY_BAND,
    _strategy_horizon_hours,
    _future_extreme_signed_returns,
    get_BIN_META,
    get_CLASS_BIN,
)

logger = logging.getLogger(__name__)

# ===== 설정: config 우선, env는 보조 =====
_BIN_META = dict(get_BIN_META() or {})
_TARGET_BINS = int(os.getenv("TARGET_BINS", str(_BIN_META.get("TARGET_BINS", 8))))
_OUT_Q_LOW = float(os.getenv("OUTLIER_Q_LOW", str(_BIN_META.get("OUTLIER_Q_LOW", 0.01))))
_OUT_Q_HIGH = float(os.getenv("OUTLIER_Q_HIGH", str(_BIN_META.get("OUTLIER_Q_HIGH", 0.99))))

# 단일 bin 폭 상한(절대 %)
_MAX_BIN_SPAN_PCT = float(os.getenv("MAX_BIN_SPAN_PCT", str(_BIN_META.get("MAX_BIN_SPAN_PCT", 8.0))))
# 최소 샘플 비율(희소 bin 병합 기준)
_MIN_BIN_COUNT_FRAC = float(os.getenv("MIN_BIN_COUNT_FRAC", str(_BIN_META.get("MIN_BIN_COUNT_FRAC", 0.05))))

# 추가 메타(지배적 bin 제어, 센터 밴드 상한)
_DOMINANT_MAX_FRAC = float(os.getenv("DOMINANT_MAX_FRAC", str(_BIN_META.get("DOMINANT_MAX_FRAC", 0.35))))
_DOMINANT_MAX_ITERS = int(os.getenv("DOMINANT_MAX_ITERS", str(_BIN_META.get("DOMINANT_MAX_ITERS", 6))))

# 중앙 밴드 상한(%) — 기본 0.5%
_CENTER_SPAN_MAX_PCT = float(os.getenv("CENTER_SPAN_MAX_PCT", str(_BIN_META.get("CENTER_SPAN_MAX_PCT", 0.5))))

# CLASS_BIN zero-band 힌트
_CLASS_BIN_META: Dict = dict(get_CLASS_BIN() or {})
_ZERO_BAND_PCT_HINT = float(_CLASS_BIN_META.get("ZERO_BAND_PCT", _CENTER_SPAN_MAX_PCT))

# === [NEW] 퍼시스턴트 저장소(엣지 고정) ===
_EDGES_DIR = Path(os.getenv("LABEL_EDGES_DIR", "/persistent/label_edges")).resolve()
_EDGES_DIR.mkdir(parents=True, exist_ok=True)

# === [NEW] 퍼센트/비율 자동 보정 ===
def _as_ratio(x: float) -> float:
    """값이 1.0 이상이면 퍼센트(%)로 보고 100으로 나눠 비율로 변환."""
    try:
        xv = float(x)
    except Exception:
        return 0.0
    return xv / 100.0 if xv >= 1.0 else xv

# 아주 작은 수익률은 학습 제외(기본 ±0.3%)
_RAW_MIN_GAIN_FOR_TRAIN = float(os.getenv("MIN_GAIN_FOR_TRAIN", "0.003"))
_MIN_GAIN_FOR_TRAIN = _as_ratio(_RAW_MIN_GAIN_FOR_TRAIN)

# 라벨 안정화 상수
_MIN_CLASS_FRAC = 0.01
_MIN_CLASS_ABS = 8
_Q_EPS = 1e-9
_EDGE_EPS = 1e-12

# 커버리지 회복을 위한 동적 완화 한계
_RAW_MIN_GAIN_LOWER_BOUND = float(os.getenv("MIN_GAIN_LOWER_BOUND", "0.0005"))  # 5bp 혹은 0.05%로 들어와도 자동 보정
_MIN_GAIN_LOWER_BOUND = _as_ratio(_RAW_MIN_GAIN_LOWER_BOUND)

# 미세 잡음
_JITTER_EPS = float(os.getenv("GAINS_JITTER_EPS", "1e-8"))

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
    if h <= 4: return "단기"
    if h <= 24: return "중기"
    return "장기"


# -----------------------------
# Target construction
# -----------------------------
def signed_future_return_by_hours(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
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
        logger.warning("labels: _future_extreme_signed_returns returned invalid size for h=%s", horizon_hours)
        return np.zeros(n, dtype=np.float32)

    dn = both[:n]
    up = both[n:]
    gains = np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)

    # 완전 상수열/저분산 보호: 미세 잡음 부여
    if gains.size and (np.allclose(gains, gains[0]) or np.nanstd(gains) < 1e-10):
        idx = np.arange(n, dtype=np.float32)
        gains = gains + (idx - idx.mean()) * _JITTER_EPS
        logger.info("labels: gains variance tiny (h=%s) -> jitter injected", horizon_hours)
    return gains


def signed_future_return(df: pd.DataFrame, strategy: str) -> np.ndarray:
    horizon_hours = _strategy_horizon_hours(strategy)
    return signed_future_return_by_hours(df, horizon_hours=horizon_hours)


# -----------------------------
# Helpers
# -----------------------------
def _vector_bin(gains: np.ndarray, edges: np.ndarray) -> np.ndarray:
    e = edges.astype(float).copy()
    e[-1] = e[-1] + _EDGE_EPS
    bins = np.searchsorted(e, gains, side="right") - 1
    return np.clip(bins, 0, edges.size - 2).astype(np.int64)

def _coverage(x: np.ndarray) -> int:
    v = x[x >= 0]
    return int(np.unique(v).size) if v.size > 0 else 0

def _needs_rebin(labels: np.ndarray, k: int, n: int) -> bool:
    if n == 0: return False
    req = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * n)))
    vals, cnts = np.unique(labels[labels >= 0], return_counts=True) if (labels >= 0).any() else (np.array([]), np.array([]))
    if vals.size <= 2: return True
    return bool((cnts < req).any())

def _clip_outliers(g: np.ndarray) -> Tuple[np.ndarray, float, float]:
    low = np.quantile(g, _OUT_Q_LOW)
    high = np.quantile(g, _OUT_Q_HIGH)
    if not np.isfinite(low): low = np.min(g)
    if not np.isfinite(high): high = np.max(g)
    return np.clip(g, low, high), float(low), float(high)

def _dedupe_edges(edges: np.ndarray) -> np.ndarray:
    e = edges.astype(float).copy()
    for i in range(1, e.size):
        if not np.isfinite(e[i]): e[i] = e[i-1] + _Q_EPS
        if e[i] <= e[i-1]: e[i] = e[i-1] + _Q_EPS
    return e

def _equal_freq_edges(g: np.ndarray, k: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, k + 1)
    cuts = np.quantile(g, qs)
    return _dedupe_edges(cuts)

def _split_wide_bins(edges: np.ndarray, max_span_pct: float) -> np.ndarray:
    max_span = max_span_pct / 100.0
    e = edges.tolist()
    i = 0
    while i < len(e) - 1:
        lo, hi = float(e[i]), float(e[i+1])
        span = abs(hi - lo)
        if span > max_span:
            m = int(np.ceil(span / max_span))
            sub = np.linspace(lo, hi, m + 1).tolist()
            e = e[:i] + sub + e[i+2:]
            i += m
        else:
            i += 1
    return _dedupe_edges(np.array(e, dtype=float))

def _merge_sparse_bins(edges: np.ndarray, counts: np.ndarray, min_count: int) -> Tuple[np.ndarray, np.ndarray]:
    e = edges.astype(float).tolist()
    c = counts.astype(int).tolist()
    changed = True
    while changed and len(e) > 2:
        changed = False
        for i in range(len(c)):
            if c[i] < min_count:
                if i == 0:
                    c[i+1] += c[i]; del c[i]; del e[i+1]; changed = True; break
                elif i == len(c) - 1:
                    c[i-1] += c[i]; del c[i]; del e[i]; changed = True; break
                else:
                    if c[i-1] >= c[i+1]:
                        c[i-1] += c[i]; del c[i]; del e[i]; changed = True; break
                    else:
                        c[i+1] += c[i]; del c[i]; del e[i+1]; changed = True; break
    return np.array(e, dtype=float), np.array(c, dtype=int)

def _enforce_zero_band(edges: np.ndarray, zero_band_pct: float) -> np.ndarray:
    if edges.size < 3:
        return edges.astype(float)
    e = edges.astype(float).copy()
    zmax = max(0.0, float(zero_band_pct)) / 100.0
    for i in range(e.size - 1):
        lo, hi = float(e[i]), float(e[i+1])
        if lo < 0.0 <= hi:
            span = hi - lo
            if zmax > 0 and span > zmax:
                m = int(np.ceil(span / zmax))
                sub = np.linspace(lo, hi, m + 1)
                e = np.concatenate([e[:i], sub, e[i+2:]]).astype(float)
            break
    return _dedupe_edges(e)

def _ensure_zero_edge(edges: np.ndarray) -> np.ndarray:
    """양/음 값을 모두 포함하면 경계에 0을 반드시 삽입."""
    e = edges.astype(float).copy()
    if e.size < 2: return e
    if np.min(e) < 0.0 and np.max(e) > 0.0:
        if not np.any(np.isclose(e, 0.0, atol=_Q_EPS)):
            if e[0] < 0.0 < e[-1]:
                e = np.sort(np.append(e, 0.0)).astype(float)
    return _dedupe_edges(e)

def _limit_dominant_bins(edges: np.ndarray, x_clip: np.ndarray,
                         max_frac: float, max_iters: int,
                         center_span_max_pct: float) -> np.ndarray:
    if edges.size < 3 or x_clip.size == 0:
        return edges.astype(float)

    it = 0
    e = edges.astype(float).copy()
    center_max = max(0.0, center_span_max_pct) / 100.0

    while it < max_iters:
        # 중앙 구간 폭 제한(선적용)
        changed_center = False
        for i in range(e.size - 1):
            lo, hi = float(e[i]), float(e[i+1])
            if lo < 0.0 <= hi:
                span = hi - lo
                if center_max > 0 and span > center_max:
                    m = int(np.ceil(span / center_max))
                    sub = np.linspace(lo, hi, m + 1)
                    e = np.concatenate([e[:i], sub, e[i+2:]]).astype(float)
                    changed_center = True
                break
        if changed_center:
            it += 1
            continue

        # 카운트 기반 지배적 bin 탐지
        e_cnt = e.copy(); e_cnt[-1] += _EDGE_EPS
        counts, _ = np.histogram(x_clip, bins=e_cnt)
        n = int(x_clip.size)
        if n <= 0: break
        fracs = counts / float(n)
        worst_idx = int(np.argmax(fracs)) if fracs.size > 0 else -1
        worst_frac = float(fracs[worst_idx]) if worst_idx >= 0 else 0.0

        if worst_frac <= max_frac:
            break

        i = worst_idx
        lo, hi = float(e[i]), float(e[i+1])
        sub = x_clip[(x_clip >= lo) & (x_clip <= hi)]
        if sub.size < 4 or not np.isfinite(sub).any():
            mid = (lo + hi) / 2.0
        else:
            mid = float(np.quantile(sub, 0.5))
        if not np.isfinite(mid) or mid <= lo or mid >= hi:
            mid = (lo + hi) / 2.0

        e = np.insert(e, i + 1, mid).astype(float)
        it += 1

    return _dedupe_edges(e)


def _build_bins(gains: np.ndarray, target_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(gains, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        edges = np.array([-0.05, -0.02, -0.005, 0.005, 0.02, 0.05], dtype=float)
        counts = np.zeros(edges.size - 1, dtype=int)
        spans = np.diff(edges) * 100.0
        return edges, counts, spans

    x_clip, _, _ = _clip_outliers(x)
    k = max(2, int(target_bins))
    edges = _equal_freq_edges(x_clip, k)

    # 과폭 분할
    edges = _split_wide_bins(edges, _MAX_BIN_SPAN_PCT)

    # 카운트
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    counts, _ = np.histogram(x_clip, bins=edges_count)

    # 희소 병합
    min_count = max(1, int(np.ceil(_MIN_BIN_COUNT_FRAC * x_clip.size)))
    edges, counts = _merge_sparse_bins(edges, counts, min_count)

    # 지배적 분할 + 중앙 폭 제한
    edges = _limit_dominant_bins(
        edges, x_clip,
        max_frac=float(_DOMINANT_MAX_FRAC),
        max_iters=int(_DOMINANT_MAX_ITERS),
        center_span_max_pct=float(_CENTER_SPAN_MAX_PCT),
    )

    # zero-band 초미세 강제 + 0 경계 보장
    edges = _enforce_zero_band(edges, _ZERO_BAND_PCT_HINT)
    edges = _ensure_zero_edge(edges)

    spans_pct = np.diff(edges) * 100.0
    return edges.astype(float), counts.astype(int), spans_pct.astype(float)


# ============================
# [NEW] Stable Edge Store I/O
# ============================
def _edge_key(symbol: str, strategy: str) -> str:
    return f"{symbol.strip().upper()}__{strategy.strip()}"

def _edge_path(symbol: str, strategy: str) -> Path:
    key = _edge_key(symbol, strategy)
    return _EDGES_DIR / f"{key}.json"

def _hash_array(a: np.ndarray) -> str:
    try:
        b = np.ascontiguousarray(a.astype(np.float64)).tobytes()
        return hashlib.md5(b).hexdigest()
    except Exception:
        return "na"

def _save_edges(symbol: str, strategy: str, edges: np.ndarray, meta: dict) -> None:
    p = _edge_path(symbol, strategy)
    data = {
        "symbol": symbol,
        "strategy": strategy,
        "edges": list(map(float, edges.tolist())),
        "edges_hash": _hash_array(edges),
        "meta": meta or {},
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("labels: edges saved -> %s (%s/%s)", str(p), symbol, strategy)
    except Exception as e:
        logger.warning("labels: failed to save edges (%s): %s", str(p), e)

def _load_edges(symbol: str, strategy: str) -> np.ndarray | None:
    p = _edge_path(symbol, strategy)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        e = np.array(data.get("edges", []), dtype=float)
        if e.size >= 2:
            return _dedupe_edges(e)
    except Exception as e:
        logger.warning("labels: failed to load edges (%s): %s", str(p), e)
    return None


# ============================
# Core binning with boundary mask + ADAPTIVE REBIN
# ============================
def _bin_with_boundary_mask(
    gains: np.ndarray,
    edges: np.ndarray,
    symbol: str,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = gains.shape[0]
    if edges is None or edges.size < 2:
        logger.warning("labels: empty edges for %s/%s -> safe fallback", symbol, strategy)
        safe_edges = np.array([-1e-6, 0.0, 1e-6], dtype=float)
        return np.full(n, -1, dtype=np.int64), safe_edges

    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)

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
    def _masked_ratio(x: np.ndarray) -> float:
        return float((x == -1).sum()) / float(max(1, x.size))

    for shrink in (0.5, 0.25):
        if _masked_ratio(labels) > 0.60:
            cand = _apply_mask(float(BOUNDARY_BAND) * shrink)
            if _masked_ratio(cand) < _masked_ratio(labels):
                labels = cand
                logger.info("labels: mask ratio reduced to %.4f (%s/%s)", float(BOUNDARY_BAND) * shrink, symbol, strategy)

    # 4) 커버리지 점검. 최악이면 마스킹 해제
    uniq = _coverage(labels)
    if uniq <= 1:
        labels = bins.copy()
        uniq = _coverage(labels)
        logger.warning("labels: boundary mask disabled to recover coverage (uniq=%d) %s/%s", uniq, symbol, strategy)

    # 5) ADAPTIVE REBIN (동일 세션 내 최종 복구용)
    k = edges.size - 1
    n = gains.size
    req_min = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * max(1, n))))
    trigger = (uniq <= 2) or _needs_rebin(labels, k, n)
    if trigger and k >= 3 and n > 0:
        try:
            # 5.1 동일 k로 재분할
            edges2, _, _ = _build_bins(gains, k)
            labels_dyn = _vector_bin(gains, edges2)
            uniq_dyn = _coverage(labels_dyn)
            _, cnts = np.unique(labels_dyn, return_counts=True)
            ok_min = (cnts[cnts > 0].min() >= req_min) if cnts.size > 0 else False
            if (uniq_dyn > uniq) or ok_min:
                logger.info("labels: ADAPTIVE_REBIN applied (uniq %d→%d, min_req=%d) %s/%s",
                            uniq, uniq_dyn, req_min, symbol, strategy)
                return labels_dyn.astype(np.int64), edges2.astype(float)

            # 5.2 k 축소 폴백
            for k2 in range(max(3, k - 1), 2, -1):
                edges3, _, _ = _build_bins(gains, k2)
                labels_dyn2 = _vector_bin(gains, edges3)
                uniq_dyn2 = _coverage(labels_dyn2)
                _, cnts2 = np.unique(labels_dyn2, return_counts=True)
                ok_min2 = (cnts2[cnts2 > 0].min() >= req_min) if cnts2.size > 0 else False
                if uniq_dyn2 >= 3 and ok_min2:
                    logger.info("labels: FALLBACK_REBIN (k=%d) applied (uniq=%d, min_ok=%s) %s/%s",
                                k2, uniq_dyn2, ok_min2, symbol, strategy)
                    return labels_dyn2.astype(np.int64), edges3.astype(float)
        except Exception as e:
            logger.warning("labels: adaptive rebin failed: %s", e)

    return labels.astype(np.int64), edges.astype(float)


# ----- 최종 안전장치: 최소 2~3 클래스 보장(+ sign-aware 단계) -------------
def _sign_aware_rebin(gains: np.ndarray, target_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    양/음 데이터를 분리하여 각자 분위수로 나눈 후 0을 경계로 결합.
    양/음이 모두 존재하면 최소 3클래스(음/0/양) 이상 확보.
    """
    x = np.asarray(gains, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        e = np.array([-1e-6, 0.0, 1e-6], dtype=float)
        l = _vector_bin(gains, e)
        return l, e

    neg = x[x < 0.0]
    pos = x[x > 0.0]
    has_neg, has_pos = neg.size > 0, pos.size > 0

    if not (has_neg or has_pos):
        e = np.array([-1e-6, 0.0, 1e-6], dtype=float)
        l = _vector_bin(gains, e)
        return l, e

    # 목표 bin을 양/음에 나눠 할당 (최소 1개씩)
    k_total = max(3, int(target_bins))
    k_side = max(1, (k_total - 1) // 2)  # 중앙(0) 포함 가정
    edges_parts = []

    if has_neg:
        k_n = min(k_side + 1, max(2, min(6, neg.size)))  # 에지 개수 = bin+1
        qs_n = np.linspace(0.0, 1.0, k_n)
        e_n = np.quantile(neg, qs_n)
        edges_parts.append(np.unique(e_n))
    else:
        edges_parts.append(np.array([-1e-6], dtype=float))

    # 0 경계
    edges_parts.append(np.array([0.0], dtype=float))

    if has_pos:
        k_p = min(k_side + 1, max(2, min(6, pos.size)))
        qs_p = np.linspace(0.0, 1.0, k_p)
        e_p = np.quantile(pos, qs_p)
        edges_parts.append(np.unique(e_p))
    else:
        edges_parts.append(np.array([1e-6], dtype=float))

    e = np.concatenate(edges_parts).astype(float)
    e = np.unique(e)
    if e[0] == 0.0:  # 0이 맨 앞에 오지 않게 보정
        e = np.insert(e, 0, e[0] - _Q_EPS)
    if e[-1] == 0.0:
        e = np.append(e, e[-1] + _Q_EPS)

    e = _dedupe_edges(e)
    if e.size < 3:
        e = np.array([-1e-6, 0.0, 1e-6], dtype=float)

    l = _vector_bin(gains, e)
    return l.astype(np.int64), e.astype(float)


def _ensure_min_two_classes(gains: np.ndarray, labels: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    small mask 등으로 한 클래스만 남는 경우를 최종 복구.
    1) 전체 표본으로 재분할 시도
    2) sign-aware rebin(양/음 분리 분위수)
    3) 그래도 실패하면 sign 기반 2분할 폴백
    """
    cov = _coverage(labels)
    if cov >= 2:
        return labels, edges

    # 1) 전체 표본으로 재분할
    try:
        e2, _, _ = _build_bins(gains, max(2, min(8, edges.size - 1)))
        l2 = _vector_bin(gains, e2)
        if _coverage(l2) >= 2:
            logger.warning("labels: final recovery by rebuild (classes=%d)", _coverage(l2))
            return l2.astype(np.int64), e2.astype(float)
    except Exception:
        pass

    # 2) sign-aware rebin
    try:
        l3, e3 = _sign_aware_rebin(gains, target_bins=max(4, _TARGET_BINS))
        if _coverage(l3) >= 2:
            logger.warning("labels: sign-aware rebin applied (classes=%d)", _coverage(l3))
            return l3.astype(np.int64), e3.astype(float)
    except Exception as e:
        logger.warning("labels: sign-aware rebin failed: %s", e)

    # 3) sign 기반 2분할 폴백
    eps = 1e-6
    e4 = np.array([-eps, 0.0, eps], dtype=float)
    l4 = _vector_bin(gains, e4)
    logger.warning("labels: final emergency fallback to sign split (2 classes)")
    return l4.astype(np.int64), e4.astype(float)


# ============================
# [NEW] Stable edges selector
# ============================
def _get_stable_edges_for(symbol: str, strategy: str, gains_for_edges: np.ndarray) -> np.ndarray:
    """
    1) 저장된 엣지가 있으면 그대로 사용
    2) 없으면 현재 표본으로 생성 후 저장
    3) 저장된 엣지로 커버리지/희소성 문제가 심하면 '1회' 재계산 후 저장 갱신
    """
    # 1) load
    stored = _load_edges(symbol, strategy)
    if stored is not None and stored.size >= 2:
        return stored

    # 2) build & save
    edges, _, _ = _build_bins(gains_for_edges, _TARGET_BINS)
    if np.allclose(np.diff(edges), 0, atol=1e-9):
        edges = np.array([-1e-6, 0.0, 1e-6], dtype=float)
        logger.warning("labels: edges collapsed at first build → sign-split")
    _save_edges(symbol, strategy, edges, meta={"reason": "first_build"})
    return edges


def _maybe_refresh_edges(symbol: str, strategy: str,
                         gains: np.ndarray, labels: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    저장 엣지 사용 후 라벨링 결과가 너무 나쁘면 1회 재계산하고 저장 갱신.
    """
    k = edges.size - 1
    n = int(gains.size)
    if n <= 0 or k < 2:
        return edges

    req_min = max(_MIN_CLASS_ABS, int(np.ceil(_MIN_CLASS_FRAC * n)))
    cov = _coverage(labels)
    vals, cnts = np.unique(labels[labels >= 0], return_counts=True) if (labels >= 0).any() else (np.array([]), np.array([]))
    min_ok = (cnts.min() >= req_min) if cnts.size > 0 else False

    if cov >= 3 and min_ok:
        return edges  # 충분히 양호

    # 1회 재계산
    edges2, _, _ = _build_bins(gains, max(3, _TARGET_BINS))
    if np.allclose(np.diff(edges2), 0, atol=1e-9):
        edges2 = np.array([-1e-6, 0.0, 1e-6], dtype=float)

    labels2 = _vector_bin(gains, edges2)
    cov2 = _coverage(labels2)
    vals2, cnts2 = np.unique(labels2[labels2 >= 0], return_counts=True) if (labels2 >= 0).any() else (np.array([]), np.array([]))
    min_ok2 = (cnts2.min() >= req_min) if cnts2.size > 0 else False

    if (cov2 > cov) or min_ok2:
        _save_edges(symbol, strategy, edges2, meta={"reason": "refresh"})
        logger.info("labels: edges refreshed (%s/%s) cov %d→%d", symbol, strategy, cov, cov2)
        return edges2

    return edges


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
        labels:      int64   (N,)  (0..C-1)  # -1 없음
        class_ranges:List[(lo, hi)]
        bin_edges:   float64 (C+1,)
        bin_counts:  int64   (C,)
        bin_spans:   float64 (C,)  # 절대 %
    """
    gains = signed_future_return(df, strategy)  # (N,)

    # 항상 '현재 표본'으로 엣지 재계산 (저장 불사용)
    edges, _, _ = _build_bins(gains, _TARGET_BINS)

    # 경계 마스킹/(-1) 없이 전 표본 라벨링
    labels = _vector_bin(gains, edges)

    # class_ranges / counts / spans 계산
    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size - 1)]
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    bin_counts, _ = np.histogram(np.clip(gains, edges[0], edges[-1]), bins=edges_count)
    bin_spans = np.diff(edges) * 100.0

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges, edges.astype(float), bin_counts.astype(int), bin_spans.astype(float)


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
        (labels: 0..C-1, -1 없음)
    """
    gains = signed_future_return_by_hours(df, horizon_hours=int(horizon_hours))
    strategy = _strategy_from_hours(int(horizon_hours))

    # 항상 '현재 표본'으로 엣지 재계산
    edges, _, _ = _build_bins(gains, _TARGET_BINS)

    # 경계 마스킹/(-1) 없이 전 표본 라벨링
    labels = _vector_bin(gains, edges)

    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size - 1)]
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    bin_counts, _ = np.histogram(np.clip(gains, edges[0], edges[-1]), bins=edges_count)
    bin_spans = np.diff(edges) * 100.0

    return gains.astype(np.float32), labels.astype(np.int64), class_ranges, strategy, edges.astype(float), bin_counts.astype(int), bin_spans.astype(float)


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
        gains, labels, ranges, strategy, edges, counts, spans = make_labels_for_horizon(df, symbol, h, group_id=group_id)
        key = f"{h}h" if h < 24 else ("1d" if h == 24 else (f"{h//24}d" if h < 168 else "7d"))
        out[key] = (gains, labels, ranges, edges, counts, spans)
        logger.debug("make_all_horizon_labels: %s -> strategy=%s, key=%s", h, strategy, key)
    return out
