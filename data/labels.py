# ================================================
# labels.py — YOPO RAW 기반 수익률 라벨링
#  - 전략별 H 고정 (단기/중기/장기 = 각 1캔들)
#  - C0~C5 고정 6클래스 구조 적용
# ================================================
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

# --------------------------------------------
# 공통 설정 (config와 동기화)
# --------------------------------------------
_BIN_META = dict(get_BIN_META() or {})

def _as_ratio(x: float) -> float:
    try:
        xv = float(x)
    except Exception:
        return 0.0
    return xv / 100.0 if xv >= 1.0 else xv

def _as_percent(x: float) -> float:
    try:
        xv = float(x)
    except Exception:
        return 0.0
    return xv * 100.0 if 0.0 < xv < 1.0 else xv

# ✅ config.BIN_META에서 넘어온 목표 bin 수 (기본 6개)
_TARGET_BINS = int(os.getenv("TARGET_BINS", str(_BIN_META.get("TARGET_BINS", 6))))
# ✅ 최소 클래스 개수도 BIN_META에 있을 경우 따라가고, 없으면 4
_MIN_LABEL_CLASSES = int(
    os.getenv(
        "MIN_LABEL_CLASSES",
        str(_BIN_META.get("MIN_LABEL_CLASSES", 4))
    )
)

# ✅ 한 클래스당 최소 패턴 수(샘플 수) 기준
_MIN_SAMPLES_PER_CLASS = int(os.getenv("MIN_SAMPLES_PER_CLASS", "50"))

# 🔥 전략별 H 고정 (핵심 수정)
# - 단기: 4h 캔들 1개
# - 중기: 1d 캔들 1개
# - 장기: 주봉(1w) 캔들 1개
_FIXED_H = {
    "단기": 1,   # 4시간봉 → t+1캔들
    "중기": 1,   # 1일봉 → t+1캔들
    "장기": 1,   # 주봉(1w) → t+1캔들
}

_DEFAULT_STRATEGY_HOURS = {
    "단기": 4,
    "중기": 24,
    "장기": 24 * 7,
}

def _ensure_dir_with_fallback(primary: str, fallback: str) -> Path:
    p_primary = Path(primary).resolve()
    try:
        p_primary.mkdir(parents=True, exist_ok=True)
        return p_primary
    except Exception:
        p_fallback = Path(fallback).resolve()
        p_fallback.mkdir(parents=True, exist_ok=True)
        return p_fallback

_PERSIST_BASE = os.getenv("PERSIST_DIR", "/persistent")

_EDGES_DIR = _ensure_dir_with_fallback(
    os.getenv("LABEL_EDGES_DIR", f"{_PERSIST_BASE}/label_edges"),
    "/tmp/label_edges",
)
_LABELS_DIR = _ensure_dir_with_fallback(
    os.getenv("LABEL_TABLE_DIR", f"{_PERSIST_BASE}/labels"),
    "/tmp/labels",
)

_RAW_MIN_GAIN_FOR_TRAIN = float(os.getenv("MIN_GAIN_FOR_TRAIN", "0.003"))
_MIN_GAIN_FOR_TRAIN = _as_ratio(_RAW_MIN_GAIN_FOR_TRAIN)

# ============================================================
# 시간/전략 헬퍼
# ============================================================
def _to_series_ts_kst(ts_like) -> pd.Series:
    ts = pd.to_datetime(ts_like, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    return ts

def _normalize_strategy_name(strategy: str) -> str:
    s = str(strategy).strip()
    if "단기" in s: return "단기"
    if "중기" in s: return "중기"
    if "장기" in s: return "장기"
    return s

# 🔥 (핵심수정) 전략별 H를 무조건 고정값으로 반환
def _get_fixed_horizon_candles(strategy: str) -> int:
    pure = _normalize_strategy_name(strategy)
    return int(_FIXED_H.get(pure, 1))

# ============================================================
# 미래 수익률(H개 캔들 동안 high/low)
# ============================================================
def _future_extreme_signed_returns_by_candles(df: pd.DataFrame, H: int):
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float32)
    high  = pd.to_numeric(df.get("high", df["close"]), errors="coerce").to_numpy(dtype=np.float32)
    low   = pd.to_numeric(df.get("low",  df["close"]), errors="coerce").to_numpy(dtype=np.float32)

    up = np.zeros(n, dtype=np.float32)
    dn = np.zeros(n, dtype=np.float32)
    H = max(1, int(H))

    for i in range(n):
        j = min(n, i + H)
        base = close[i] if close[i] > 0 else 1e-6
        up[i] = (float(np.max(high[i:j])) - base) / (base + 1e-12)
        dn[i] = (float(np.min(low[i:j])) - base) / (base + 1e-12)

    return up, dn

# ============================================================
# RAW gain 선택
# ============================================================
def _pick_per_candle_gain(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    return np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)

# ============================================================
# (참고용) RAW bin 생성 — 동적 버전 (지금은 안 씀)
# ============================================================
def _raw_bins(dist: np.ndarray, target_bins: int) -> np.ndarray:
    """
    [참고] 예전 동적 분포 기반 엣지 계산 함수.
    지금은 C0~C5 고정 6클래스를 쓰기 때문에
    실제 make_labels에서는 이 함수를 사용하지 않는다.
    """
    if dist is None:
        return np.linspace(-0.01, 0.01, target_bins + 1).astype(float)

    dist = np.asarray(dist, dtype=float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return np.linspace(-0.01, 0.01, target_bins + 1).astype(float)

    n = dist.size
    lo = float(np.min(dist))
    hi = float(np.max(dist))
    if not np.isfinite(lo):
        lo = -0.01
    if not np.isfinite(hi):
        hi = 0.01
    if hi <= lo:
        hi = lo + 1e-6

    if n < _MIN_SAMPLES_PER_CLASS * 2:
        bins_small = max(_MIN_LABEL_CLASSES, min(target_bins, 4))
        return np.linspace(lo, hi, bins_small + 1).astype(float)

    max_bins_by_samples = max(1, n // max(1, _MIN_SAMPLES_PER_CLASS))
    bins = int(min(target_bins, _TARGET_BINS, max_bins_by_samples))
    bins = int(max(_MIN_LABEL_CLASSES, bins))

    if bins <= 1:
        return np.linspace(lo, hi, 2 + 1).astype(float)

    qs = np.linspace(0.0, 1.0, bins + 1)
    try:
        edges = np.quantile(dist, qs)
    except Exception:
        return np.linspace(lo, hi, bins + 1).astype(float)

    edges = np.asarray(edges, dtype=float)

    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    edges[0] = min(edges[0], lo)
    edges[-1] = max(edges[-1], hi)

    return edges

def _vector_bin(gains: np.ndarray, edges: np.ndarray) -> np.ndarray:
    e = edges.copy()
    e[-1] += 1e-12
    bins = np.searchsorted(e, gains, side="right") - 1
    return np.clip(bins, 0, edges.size - 2).astype(np.int64)

# ============================================================
# target bin 수 (지금은 6고정 구조라 참고용)
# ============================================================
def _auto_target_bins(df_len: int) -> int:
    if df_len <= 0:
        return _MIN_LABEL_CLASSES

    if df_len < _MIN_SAMPLES_PER_CLASS * 2:
        return max(_MIN_LABEL_CLASSES, 2)

    max_bins_by_samples = max(1, df_len // max(1, _MIN_SAMPLES_PER_CLASS))
    bins = min(max_bins_by_samples, _TARGET_BINS)
    bins = max(_MIN_LABEL_CLASSES, bins)
    return int(bins)

# ============================================================
# 전략별 고정 6클래스 경계 정의 (C0~C5)
# ============================================================
# C0: 큰 손실
# C1: 보통 손실
# C2: 애매(소폭 손실~소폭 수익)
# C3: 보통 수익
# C4: 큰 수익
# C5: 아주 큰 수익
_FIXED_CLASS_THRESHOLDS = {
    # 단기: 변동성 작다고 보고 좁게
    "단기": np.array([-0.06, -0.02, 0.02, 0.05, 0.10], dtype=float),
    # 중기: 변동 더 크다고 보고 약간 넓게
    "중기": np.array([-0.08, -0.03, 0.03, 0.07, 0.15], dtype=float),
    # 장기: 변동 가장 크다고 보고 더 넓게
    "장기": np.array([-0.10, -0.04, 0.04, 0.10, 0.25], dtype=float),
}

def _fixed_edges_for_strategy(dist: np.ndarray, strategy: str) -> np.ndarray:
    """
    전략별로 C0~C5 고정 6클래스 경계를 만든다.
    - dist의 실제 min/max를 보고 양 끝을 살짝 확장
    - 중간 5개 경계는 _FIXED_CLASS_THRESHOLDS를 그대로 사용
    """
    pure = _normalize_strategy_name(strategy)
    th = _FIXED_CLASS_THRESHOLDS.get(pure, _FIXED_CLASS_THRESHOLDS["중기"])

    if dist is None or dist.size == 0:
        lo = min(th[0] * 1.5, -0.01)
        hi = max(th[-1] * 1.5, 0.01)
    else:
        dist = np.asarray(dist, dtype=float)
        dist = dist[np.isfinite(dist)]
        if dist.size == 0:
            lo = min(th[0] * 1.5, -0.01)
            hi = max(th[-1] * 1.5, 0.01)
        else:
            lo = float(np.min(dist))
            hi = float(np.max(dist))
            if not np.isfinite(lo):
                lo = min(th[0] * 1.5, -0.01)
            if not np.isfinite(hi):
                hi = max(th[-1] * 1.5, 0.01)

    lo = min(lo, th[0] - 1e-6)
    hi = max(hi, th[-1] + 1e-6)

    edges = np.concatenate(([lo], th, [hi])).astype(float)

    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    return edges

# ============================================================
# 수익률 계산 (H 고정)
# ============================================================
def compute_label_returns(df: pd.DataFrame, symbol: str, strategy: str):
    pure = _normalize_strategy_name(strategy)
    H = _get_fixed_horizon_candles(pure)
    up, dn = _future_extreme_signed_returns_by_candles(df, H)
    gains = _pick_per_candle_gain(up, dn)
    # target_bins는 이제 6클래스 구조에서 크게 의미 없지만, 메타 정보용으로 유지
    target = 6
    return gains, up, dn, target

# ============================================================
# 저장 헬퍼들 (그대로 유지)
# ============================================================
def _edge_key(symbol: str, strategy: str) -> str:
    return f"{symbol.upper()}__{_normalize_strategy_name(strategy)}"

def _edge_path(symbol: str, strategy: str) -> Path:
    return _EDGES_DIR / f"{_edge_key(symbol, strategy)}.json"

def _hash_array(a: np.ndarray) -> str:
    try:
        b = np.ascontiguousarray(a.astype(np.float64)).tobytes()
        return hashlib.md5(b).hexdigest()
    except:
        return "na"

def _save_edges(symbol, strategy, edges, meta):
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
    except Exception as e:
        logger.warning("failed to save edges: %s", e)

def _labels_path(symbol, strategy):
    return _LABELS_DIR / f"{_edge_key(symbol, strategy)}.parquet"

def _labels_csv_path(symbol, strategy):
    return _LABELS_DIR / f"{_edge_key(symbol, strategy)}.csv"

# ============================================================
# 라벨 저장
# ============================================================
def _save_label_table(df, symbol, strategy, gains, labels, edges, counts,
                      spans, extra_cols=None, extra_meta=None, group_id=None):

    pure = _normalize_strategy_name(strategy)
    ts = _to_series_ts_kst(df["timestamp"]) if "timestamp" in df else pd.Series(pd.NaT)

    out = pd.DataFrame({
        "ts": ts,
        "symbol": symbol,
        "strategy": pure,
        "class_id": labels,
        "signed_gain": gains,
    })
    if group_id is not None:
        out["group_id"] = int(group_id)
    if extra_cols:
        for k, v in extra_cols.items():
            out[k] = v

    p_parquet = _labels_path(symbol, pure)
    p_csv = _labels_csv_path(symbol, pure)
    p_parquet.parent.mkdir(parents=True, exist_ok=True)

    try:
        out.to_parquet(p_parquet, index=False)
    except Exception:
        out.to_csv(p_csv, index=False)

    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size - 1)]

    meta = {
        "symbol": symbol,
        "strategy": pure,
        "NUM_CLASSES": int(edges.size - 1),
        "edges": list(map(float, edges.tolist())),
        "edges_hash": _hash_array(edges),
        "class_ranges": class_ranges,
        "bin_counts": list(map(int, counts)),
    }
    if extra_meta:
        meta.update(extra_meta)
    if group_id is not None:
        meta["group_id"] = group_id

    _save_edges(symbol, pure, edges, meta)

# ============================================================
# make_labels — 전략별 C0~C5 고정 6클래스 라벨 생성
# ============================================================
def make_labels(df, symbol, strategy, group_id=None):
    pure = _normalize_strategy_name(strategy)

    gains, up_c, dn_c, target_bins = compute_label_returns(df, symbol, pure)

    # 🔥 dist: 고가/저가 둘 다 사용 (up+dn 전체 분포)
    dist = np.concatenate([dn_c, up_c]).astype(np.float32)

    # 🔥 핵심: 전략별 고정 C0~C5 경계 사용
    edges = _fixed_edges_for_strategy(dist, pure)

    labels = _vector_bin(gains, edges)

    edges2 = edges.copy()
    edges2[-1] += 1e-12
    bin_counts, _ = np.histogram(dist, bins=edges2)
    spans = np.diff(edges) * 100.0

    sl = 0.02
    extra_cols = {
        "future_up": up_c,
        "future_dn": dn_c,
        "up_ge_2pct": (up_c >= sl).astype(np.int8),
        "dn_le_-2pct": (dn_c <= -sl).astype(np.int8),
    }

    _save_label_table(
        df, symbol, pure,
        gains, labels,
        edges, bin_counts, spans,
        extra_cols=extra_cols,
        extra_meta={"target_bins_used": int(target_bins)},
        group_id=group_id,
    )

    try:
        num_classes = int(edges.size - 1)
        if spans.size > 0:
            span_min = float(spans.min())
            span_max = float(spans.max())
        else:
            span_min = span_max = 0.0
        logger.info(
            "[labels] %s-%s raw N=%d target_bins=%d actual_bins=%d "
            "counts=%s span_min=%.4f span_max=%.4f",
            symbol, pure, len(df), int(target_bins), num_classes,
            bin_counts.tolist(), span_min, span_max,
        )
    except Exception:
        pass

    class_ranges = [(float(edges[i]), float(edges[i+1]))
                    for i in range(edges.size - 1)]

    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        edges.astype(float),
        bin_counts.astype(int),
        spans.astype(float),
    )

# ============================================================
# make_labels_for_horizon (horizon_hours 기준 버전도 고정 6클래스 사용)
# ============================================================
def make_labels_for_horizon(df, symbol, horizon_hours, group_id=None):
    n = len(df)
    both = _future_extreme_signed_returns(df, horizon_hours=horizon_hours)
    if both is None:
        dn = np.zeros(n, dtype=np.float32)
        up = np.zeros(n, dtype=np.float32)
    else:
        dn = np.asarray(both[:n], dtype=np.float32)
        up = np.asarray(both[n:], dtype=np.float32)

    strategy = "단기" if horizon_hours <= 4 else ("중기" if horizon_hours <= 24 else "장기")

    gains = _pick_per_candle_gain(up, dn)

    # 🔥 dist: horizon 버전도 고가/저가 둘 다 사용
    dist = np.concatenate([dn, up]).astype(np.float32)

    edges = _fixed_edges_for_strategy(dist, strategy)

    labels = _vector_bin(gains, edges)

    edges2 = edges.copy()
    edges2[-1] += 1e-12
    counts, _ = np.histogram(dist, bins=edges2)
    spans = np.diff(edges) * 100.0

    extra_cols = {
        "future_up": up,
        "future_dn": dn,
        "up_ge_2pct": (up >= 0.02).astype(np.int8),
        "dn_le_-2pct": (dn <= -0.02).astype(np.int8),
    }

    _save_label_table(
        df, symbol, strategy, gains, labels,
        edges, counts, spans,
        extra_cols=extra_cols,
        extra_meta={"target_bins_used": 6},
        group_id=group_id,
    )

    try:
        num_classes = int(edges.size - 1)
        if spans.size > 0:
            span_min = float(spans.min())
            span_max = float(spans.max())
        else:
            span_min = span_max = 0.0
        logger.info(
            "[labels-h] %s-%s(h=%dh) raw N=%d target_bins=%d actual_bins=%d "
            "counts=%s span_min=%.4f span_max=%.4f",
            symbol, strategy, int(horizon_hours), len(df),
            6, num_classes,
            counts.tolist(), span_min, span_max,
        )
    except Exception:
        pass

    class_ranges = [(float(edges[i]), float(edges[i+1]))
                    for i in range(edges.size - 1)]

    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        strategy,
        edges.astype(float),
        counts.astype(int),
        spans.astype(float),
    )

# ============================================================
# make_all_horizon_labels
# ============================================================
def make_all_horizon_labels(df, symbol, horizons=None, group_id=None):
    if horizons is None:
        horizons = [4, 24, 168]

    out = {}
    for h in horizons:
        gains, labels, ranges, strat, edges, counts, spans = \
            make_labels_for_horizon(df, symbol, h, group_id)
        key = (
            f"{h}h" if h < 24 else
            ("1d" if h == 24 else
             (f"{h//24}d" if h < 168 else "7d"))
        )
        out[key] = (gains, labels, ranges, edges, counts, spans)
    return out
