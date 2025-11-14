# ================================================
# labels.py — YOPO RAW 기반 수익률 라벨링 (H 고정 정상화 버전)
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
# 공통 설정 (기존 유지)
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

_TARGET_BINS = int(os.getenv("TARGET_BINS", str(_BIN_META.get("TARGET_BINS", 8))))
_MIN_LABEL_CLASSES = int(os.getenv("MIN_LABEL_CLASSES", "4"))

# 🔥 전략별 H 고정 (핵심 수정)
_FIXED_H = {
    "단기": 1,   # 4시간봉 → t+1캔들
    "중기": 1,   # 1일봉 → t+1캔들
    "장기": 7,   # 1일봉 → t+7캔들
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
# RAW bin 생성
# ============================================================
def _raw_bins(dist: np.ndarray, target_bins: int) -> np.ndarray:
    if dist.size == 0:
        return np.linspace(-0.01, 0.01, target_bins + 1).astype(float)
    lo = float(np.min(dist))
    hi = float(np.max(dist))
    if not np.isfinite(lo): lo = -0.01
    if not np.isfinite(hi): hi = 0.01
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, target_bins + 1).astype(float)

def _vector_bin(gains: np.ndarray, edges: np.ndarray) -> np.ndarray:
    e = edges.copy()
    e[-1] += 1e-12
    bins = np.searchsorted(e, gains, side="right") - 1
    return np.clip(bins, 0, edges.size - 2).astype(np.int64)

# ============================================================
# target bin 수
# ============================================================
def _auto_target_bins(df_len: int) -> int:
    if df_len <= 300:  return max(8, _TARGET_BINS)
    if df_len <= 600:  return max(10, _TARGET_BINS)
    if df_len <= 1000: return max(14, _TARGET_BINS)
    if df_len <= 2000: return max(18, _TARGET_BINS)
    if df_len <= 4000: return max(24, _TARGET_BINS)
    return max(32, _TARGET_BINS)

# ============================================================
# 수익률 계산 (핵심 수정)
# ============================================================
def compute_label_returns(df: pd.DataFrame, symbol: str, strategy: str):
    pure = _normalize_strategy_name(strategy)
    H = _get_fixed_horizon_candles(pure)   # 🔥 전략별 H 고정 적용
    up, dn = _future_extreme_signed_returns_by_candles(df, H)
    gains = _pick_per_candle_gain(up, dn)
    target = _auto_target_bins(len(df))
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
# make_labels
# ============================================================
def make_labels(df, symbol, strategy, group_id=None):
    pure = _normalize_strategy_name(strategy)

    gains, up_c, dn_c, target_bins = compute_label_returns(df, symbol, pure)

    dist = np.concatenate([dn_c, up_c], axis=0)

    edges = _raw_bins(dist, target_bins)

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
        extra_meta={"target_bins_used": target_bins},
        group_id=group_id,
    )

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
# make_labels_for_horizon (RAW 통일)
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

    dist = np.concatenate([dn, up], axis=0)
    target_bins = _auto_target_bins(len(df))
    edges = _raw_bins(dist, target_bins)

    gains = _pick_per_candle_gain(up, dn)
    labels = _vector_bin(gains, edges)

    edges2 = edges.copy()
    edges2[-1] += 1e-12
    counts, _ = np.histogram(dist, bins=edges2)
    spans = np.diff(edges) * 100.0

    strategy = "단기" if horizon_hours <= 4 else ("중기" if horizon_hours <= 24 else "장기")

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
        extra_meta={"target_bins_used": target_bins},
        group_id=group_id,
    )

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
