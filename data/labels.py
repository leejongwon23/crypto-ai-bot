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
    _strategy_horizon_hours,  # 남겨둠: 전략별 시간(h) 설정
    _future_extreme_signed_returns,
    get_BIN_META,
    get_CLASS_BIN,
)

logger = logging.getLogger(__name__)

# ===== 설정: config 우선, env는 보조 =====
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

# 기본 목표 bin 수(환경/설정 우선), 단 실제 사용은 아래 _auto_target_bins로 동적 적용
_TARGET_BINS = int(os.getenv("TARGET_BINS", str(_BIN_META.get("TARGET_BINS", 8))))
_OUT_Q_LOW = float(os.getenv("OUTLIER_Q_LOW", str(_BIN_META.get("OUTLIER_Q_LOW", 0.01))))
_OUT_Q_HIGH = float(os.getenv("OUTLIER_Q_HIGH", str(_BIN_META.get("OUTLIER_Q_HIGH", 0.99))))
_MAX_BIN_SPAN_PCT = _as_percent(float(os.getenv("MAX_BIN_SPAN_PCT", str(_BIN_META.get("MAX_BIN_SPAN_PCT", 8.0)))))
_MIN_BIN_COUNT_FRAC = float(os.getenv("MIN_BIN_COUNT_FRAC", str(_BIN_META.get("MIN_BIN_COUNT_FRAC", 0.05))))

_DOMINANT_MAX_FRAC = float(os.getenv("DOMINANT_MAX_FRAC", str(_BIN_META.get("DOMINANT_MAX_FRAC", 0.35))))
_DOMINANT_MAX_ITERS = int(os.getenv("DOMINANT_MAX_ITERS", str(_BIN_META.get("DOMINANT_MAX_ITERS", 6))))
_CENTER_SPAN_MAX_PCT = _as_percent(float(os.getenv("CENTER_SPAN_MAX_PCT", str(_BIN_META.get("CENTER_SPAN_MAX_PCT", 0.5)))))

_CLASS_BIN_META: Dict = dict(get_CLASS_BIN() or {})
_ZERO_BAND_PCT_HINT = _as_percent(float(_CLASS_BIN_META.get("ZERO_BAND_PCT", _CENTER_SPAN_MAX_PCT)))

_MIN_LABEL_CLASSES = int(os.getenv("MIN_LABEL_CLASSES", "4"))

# 전략별 기본 horizon (시간 기준, 바 없으면 이걸 사용)
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
    except Exception as e:
        logger.warning("labels: can't create %s (%s) -> fallback to %s", primary, e, fallback)
        p_fallback = Path(fallback).resolve()
        p_fallback.mkdir(parents=True, exist_ok=True)
        return p_fallback

# NOTE: predict.py는 PERSISTENT_DIR, 여기서는 PERSIST_DIR을 쓰지만
# 기본값이 둘 다 "/persistent" 라서 경로는 일치함.
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

_MIN_CLASS_FRAC = 0.01
_MIN_CLASS_ABS = 8
_Q_EPS = 1e-9
_EDGE_EPS = 1e-12

_RAW_MIN_GAIN_LOWER_BOUND = float(os.getenv("MIN_GAIN_LOWER_BOUND", "0.0005"))
_MIN_GAIN_LOWER_BOUND = _as_ratio(_RAW_MIN_GAIN_LOWER_BOUND)

_JITTER_EPS = float(os.getenv("GAINS_JITTER_EPS", "1e-8"))

# -----------------------------
# Timezone helper
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
def _normalize_strategy_name(strategy: str) -> str:
    s = str(strategy).strip()
    if "단기" in s:
        return "단기"
    if "중기" in s:
        return "중기"
    if "장기" in s:
        return "장기"
    return s

def _infer_bar_hours_from_df(df: pd.DataFrame) -> float:
    """
    DF의 timestamp 간격을 보고 한 캔들이 몇 시간인지 추정.
    - 실패하면 1시간으로 가정.
    """
    try:
        if "timestamp" not in df.columns or len(df) < 2:
            return 1.0
        ts = _to_series_ts_kst(df["timestamp"])
        ts = ts.sort_values()
        diffs = ts.diff().dropna()
        if diffs.empty:
            return 1.0
        med = diffs.median()
        h = med.total_seconds() / 3600.0
        if not np.isfinite(h) or h <= 0:
            return 1.0
        return float(h)
    except Exception:
        return 1.0

# 전략 → "몇 캔들"을 볼지 결정 (4h/24h/168h 같은 시간 기준을 캔들 개수로 변환)
def _strategy_horizon_candles_from_hours(df: pd.DataFrame, strategy: str) -> int:
    """
    예:
    - 캔들 간격이 1h이고, 전략이 '단기'(4h)면 → 4 캔들
    - 캔들 간격이 4h이고, 전략이 '단기'(4h)면 → 1 캔들
    - 캔들 간격이 1h이고, 전략이 '장기'(168h)면 → 168 캔들
    """
    s = _normalize_strategy_name(strategy)

    # 1) 기본 horizon 시간(시간 단위)을 config에서 우선 가져오고, 없으면 DEFAULT 사용
    base_hours = _DEFAULT_STRATEGY_HOURS.get(s, _DEFAULT_STRATEGY_HOURS["단기"])
    try:
        if isinstance(_strategy_horizon_hours, dict):
            base_hours = int(_strategy_horizon_hours.get(s, base_hours))
    except Exception:
        pass

    # 2) 캔들 한 개가 몇 시간인지 추정
    bar_hours = _infer_bar_hours_from_df(df)

    # 3) "몇 캔들" 볼지 계산 (최소 1)
    try:
        H = int(round(float(base_hours) / float(bar_hours)))
    except Exception:
        H = 1
    H = max(1, H)
    return H

# 캔들 개수로 미래 up/dn 계산
def _future_extreme_signed_returns_by_candles(df: pd.DataFrame, horizon_candles: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    close = pd.to_numeric(df.get("close"), errors="coerce").to_numpy(dtype=np.float32)
    high  = pd.to_numeric(df.get("high", df.get("close")), errors="coerce").to_numpy(dtype=np.float32)
    low   = pd.to_numeric(df.get("low",  df.get("close")), errors="coerce").to_numpy(dtype=np.float32)

    up = np.zeros(n, dtype=np.float32)
    dn = np.zeros(n, dtype=np.float32)
    H = int(max(1, horizon_candles))

    for i in range(n):
        j_end = min(n, i + H)
        base = close[i] if close[i] > 0 else 1e-6
        fut_high = float(np.max(high[i:j_end]))
        fut_low  = float(np.min(low[i:j_end]))
        up[i] = (fut_high - base) / (base + 1e-12)
        dn[i] = (fut_low  - base) / (base + 1e-12)

    return up.astype(np.float32), dn.astype(np.float32)

# -----------------------------
# 분포용이랑 라벨용을 분리
# -----------------------------
def _pick_per_candle_gain(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """
    한 캔들에 실제로 찍힐 라벨용 수익률.
    여기서는 '어느 쪽으로 더 많이 갔냐'를 기준으로 하나만 고른다.
    (하지만 분포는 위에서 둘 다 넣을 거임)
    """
    gains = np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)
    gains = np.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype(np.float32)
    if gains.size and (np.allclose(gains, gains[0]) or np.nanstd(gains) < 1e-10):
        idx = np.arange(gains.size, dtype=np.float32)
        gains = gains + (idx - idx.mean()) * _JITTER_EPS
    return gains

# -----------------------------
# bin 만드는 공통 함수들
# -----------------------------
def _vector_bin(gains: np.ndarray, edges: np.ndarray) -> np.ndarray:
    e = edges.astype(float).copy()
    e[-1] = e[-1] + _EDGE_EPS
    bins = np.searchsorted(e, gains, side="right") - 1
    return np.clip(bins, 0, edges.size - 2).astype(np.int64)

def _clip_outliers(g: np.ndarray) -> Tuple[np.ndarray, float, float]:
    if g.size == 0:
        return g, 0.0, 0.0
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

def _drop_empty_bins(edges: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if edges.size < 2 or counts.size != edges.size - 1:
        return edges.astype(float), counts.astype(int) if counts is not None else np.zeros(0, dtype=int)

    keep_mask = counts.astype(int) > 0
    if np.all(keep_mask):
        return edges.astype(float), counts.astype(int)

    new_edges = [float(edges[0])]
    new_counts = []
    for i, cnt in enumerate(counts):
        if int(cnt) > 0:
            new_edges.append(float(edges[i+1]))
            new_counts.append(int(cnt))
    if len(new_edges) < 2:
        return edges.astype(float), counts.astype(int)
    return _dedupe_edges(np.array(new_edges, dtype=float)), np.array(new_counts, dtype=int)

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
    e = edges.astype(float).copy()
    if e.size < 2:
        return e
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

        e_cnt = e.copy(); e_cnt[-1] += _EDGE_EPS
        counts, _ = np.histogram(x_clip, bins=e_cnt)
        n = int(x_clip.size)
        if n <= 0:
            break
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

def _ensure_min_classes(edges: np.ndarray, x_clip: np.ndarray, min_classes: int,
                        zero_band_pct: float) -> np.ndarray:
    cur = int(max(0, edges.size - 1))
    if cur >= max(2, min_classes):
        return edges.astype(float)

    if x_clip.size == 0:
        base_edges = np.array([-0.01, -0.003, 0.003, 0.01, 0.03], dtype=float)
        if base_edges.size - 1 >= min_classes:
            return base_edges[: min_classes + 1]
        return np.linspace(-0.03, 0.03, min_classes + 1).astype(float)

    lo = float(np.min(x_clip))
    hi = float(np.max(x_clip))
    if not np.isfinite(lo):
        lo = -0.01
    if not np.isfinite(hi):
        hi = 0.01
    if hi <= lo:
        hi = lo + 0.0001

    new_edges = np.linspace(lo, hi, int(min_classes) + 1).astype(float)
    new_edges = _enforce_zero_band(new_edges, zero_band_pct)
    new_edges = _ensure_zero_edge(new_edges)
    return _dedupe_edges(new_edges)

def _build_bins(gains: np.ndarray, target_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(gains, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        edges = np.linspace(-0.03, 0.03, _MIN_LABEL_CLASSES + 1).astype(float)
        counts = np.zeros(edges.size - 1, dtype=int)
        spans = np.diff(edges) * 100.0
        return edges, counts, spans

    x_clip, _, _ = _clip_outliers(x)
    k = max(2, int(target_bins))
    edges = _equal_freq_edges(x_clip, k)
    edges = _split_wide_bins(edges, _MAX_BIN_SPAN_PCT)

    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    counts, _ = np.histogram(x_clip, bins=edges_count)

    edges, counts = _drop_empty_bins(edges, counts)

    edges = _limit_dominant_bins(
        edges, x_clip,
        max_frac=float(_DOMINANT_MAX_FRAC),
        max_iters=int(_DOMINANT_MAX_ITERS),
        center_span_max_pct=float(_CENTER_SPAN_MAX_PCT),
    )

    edges = _enforce_zero_band(edges, _ZERO_BAND_PCT_HINT)
    edges = _ensure_zero_edge(edges)

    edges = _ensure_min_classes(edges, x_clip, _MIN_LABEL_CLASSES, _ZERO_BAND_PCT_HINT)

    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    counts, _ = np.histogram(x_clip, bins=edges_count)
    spans_pct = np.diff(edges) * 100.0

    return edges.astype(float), counts.astype(int), spans_pct.astype(float)

# ============================
# Stable Edge Store I/O
# ============================
def _edge_key(symbol: str, strategy: str) -> str:
    pure_strategy = _normalize_strategy_name(strategy)
    return f"{symbol.strip().upper()}__{pure_strategy.strip()}"

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
    pure_strategy = _normalize_strategy_name(strategy)
    p = _edge_path(symbol, pure_strategy)
    data = {
        "symbol": symbol,
        "strategy": pure_strategy,
        "edges": list(map(float, edges.tolist())),
        "edges_hash": _hash_array(edges),
        "meta": meta or {},
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("labels: edges saved -> %s (%s/%s)", str(p), symbol, pure_strategy)
    except Exception as e:
        logger.warning("labels: failed to save edges (%s): %s", str(p), e)

# ============================
# Label Table I/O
# ============================
def _labels_path(symbol: str, strategy: str) -> Path:
    key = _edge_key(symbol, strategy)
    return _LABELS_DIR / f"{key}.parquet"

def _labels_csv_path(symbol: str, strategy: str) -> Path:
    key = _edge_key(symbol, strategy)
    return _LABELS_DIR / f"{key}.csv"

def _counts_dict(arr: np.ndarray) -> Dict[int, int]:
    if arr.size == 0:
        return {}
    vals, cnts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}

def _save_label_table(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    gains: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    counts: np.ndarray,
    spans_pct: np.ndarray,
    extra_cols: Dict[str, np.ndarray] | None = None,
    extra_meta: Dict[str, object] | None = None,
    group_id: int | None = None,
) -> None:
    pure_strategy = _normalize_strategy_name(strategy)

    ts = _to_series_ts_kst(df["timestamp"]) if "timestamp" in df.columns else pd.Series(pd.NaT, index=range(len(gains)))
    out = pd.DataFrame({
        "ts": ts,
        "symbol": symbol,
        "strategy": pure_strategy,
        "class_id": labels.astype(np.int64),
        "signed_gain": gains.astype(np.float32),
    })
    if group_id is not None:
        out["group_id"] = int(group_id)

    if extra_cols:
        for k, v in extra_cols.items():
            try:
                out[k] = np.asarray(v)
            except Exception:
                logger.warning("labels: failed to attach extra column '%s'", k)

    p_parquet = _labels_path(symbol, pure_strategy)
    p_csv = _labels_csv_path(symbol, pure_strategy)
    p_parquet.parent.mkdir(parents=True, exist_ok=True)

    try:
        out.to_parquet(p_parquet, index=False)
        logger.info("labels: table saved -> %s (%s/%s) rows=%d", str(p_parquet), symbol, pure_strategy, len(out))
    except Exception as e:
        try:
            out.to_csv(p_csv, index=False)
            logger.warning(
                "labels: failed to save label table (%s/%s) to Parquet (%s): %s -> fallback to CSV (%s) rows=%d",
                symbol, pure_strategy, str(p_parquet), e, str(p_csv), len(out)
            )
        except Exception as e2:
            logger.warning(
                "labels: failed to save label table (%s/%s) to CSV (%s) as well: %s",
                symbol, pure_strategy, str(p_csv), e2
            )

    num_classes = int(max(0, edges.size - 1))
    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(num_classes)]
    meta = {
        "symbol": symbol,
        "strategy": pure_strategy,
        "NUM_CLASSES": num_classes,
        "class_counts_label_freeze": _counts_dict(labels),
        "edges": list(map(float, edges.tolist())),
        "edges_hash": _hash_array(edges),
        "bin_counts": list(map(int, counts.tolist())) if counts is not None else [],
        "bin_spans_pct": list(map(float, spans_pct.tolist())) if spans_pct is not None else [],
        "dynamic_classes": True,
        "allow_trainer_class_collapse": False,
        "trainer_should_use_exact_num_classes": True,
        "candle_horizon_used": int(_strategy_horizon_candles_from_hours(df, pure_strategy)),
        "class_ranges": class_ranges,
    }
    if group_id is not None:
        meta["group_id"] = int(group_id)
    if isinstance(extra_meta, dict) and extra_meta:
        try:
            meta.update(extra_meta)
        except Exception:
            pass

    _save_edges(symbol, pure_strategy, edges, meta=meta)

# ============================
# stoploss 통계
# ============================
def _compute_class_stop_frac(
    labels: np.ndarray,
    edges: np.ndarray,
    up: np.ndarray,
    dn: np.ndarray,
    stoploss_abs: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    C = max(0, edges.size - 1)
    if C <= 0 or labels.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    class_stop_frac = np.zeros(C, dtype=float)
    class_stop_n = np.zeros(C, dtype=int)

    mids = np.array([(float(edges[i]) + float(edges[i+1])) / 2.0 for i in range(C)], dtype=float)

    up = np.asarray(up, dtype=float)
    dn = np.asarray(dn, dtype=float)
    labs = np.asarray(labels, dtype=int)

    for c in range(C):
        idx = (labs == c)
        n = int(np.sum(idx))
        class_stop_n[c] = n
        if n == 0:
            class_stop_frac[c] = 0.0
            continue
        mid = mids[c]
        if mid > 0.0:
            stop_hit = (dn[idx] <= -abs(stoploss_abs))
        elif mid < 0.0:
            stop_hit = (up[idx] >= abs(stoploss_abs))
        else:
            stop_hit = (dn[idx] <= -abs(stoploss_abs)) | (up[idx] >= abs(stoploss_abs))
        class_stop_frac[c] = float(np.mean(stop_hit.astype(np.float32)))

    return class_stop_frac, class_stop_n

# ============================================
# 🔧 데이터 길이에 따라 클래스 수 자동 조정
# ============================================
def _auto_target_bins(df_len: int) -> int:
    """
    데이터(행) 개수에 따라 사용할 target_bins를 자동 결정.
    - 최소 8, 최대 32
    - 데이터가 많을수록 더 세밀하게 분할
    - 환경변수/설정에 TARGET_BINS가 강제 지정되면 그 값이 '최소' 기준이 됨
    """
    base_min = max(8, int(_TARGET_BINS))  # 설정값이 더 크면 그걸 하한으로 삼음
    if df_len <= 300:
        return max(base_min, 8)
    elif df_len <= 600:
        return max(base_min, 10)
    elif df_len <= 1000:
        return max(base_min, 14)
    elif df_len <= 2000:
        return max(base_min, 18)
    elif df_len <= 4000:
        return max(base_min, 24)
    else:
        return max(base_min, 32)

# ============================
# Public API (strategy-based)
# ============================
def make_labels(
    df: pd.DataFrame,
    symbol: str,
    strategy: str,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
    pure_strategy = _normalize_strategy_name(strategy)

    # 1) 전략에 맞는 horizon 캔들 수 결정 (4h/24h/168h → 캔들 개수로 변환)
    horizon_candles = _strategy_horizon_candles_from_hours(df, pure_strategy)

    # 2) 해당 horizon 동안의 future up/dn 계산
    up_c, dn_c = _future_extreme_signed_returns_by_candles(df, horizon_candles)

    # 3) 분포는 dn+up 모두 넣어서 만든다
    dist_for_bins = np.concatenate([dn_c, up_c], axis=0)

    # 🔸 df 길이에 맞춰 target_bins 동적 결정
    dynamic_bins = _auto_target_bins(len(df))
    edges, bin_counts, bin_spans = _build_bins(dist_for_bins, dynamic_bins)

    # 4) 실제 라벨로는 그 캔들이 더 크게 움직인 쪽을 쓴다
    gains = _pick_per_candle_gain(up_c, dn_c)
    labels = _vector_bin(gains, edges)

    # extra columns
    extra_cols = {
        "future_up": up_c,
        "future_dn": dn_c,
    }
    sl = 0.02
    extra_cols["up_ge_2pct"] = (up_c >= sl).astype(np.int8)
    extra_cols["dn_le_-2pct"] = (dn_c <= -sl).astype(np.int8)
    extra_cols["conflict_2pct"] = ((up_c >= sl) & (dn_c <= -sl)).astype(np.int8)

    # stoploss 통계
    extra_meta = {}
    try:
        class_stop_frac, class_stop_n = _compute_class_stop_frac(
            labels,
            edges,
            up_c,
            dn_c,
            stoploss_abs=sl,
        )
        extra_meta.update({
            "stoploss_threshold_abs": float(sl),
            "class_stop_frac": list(map(float, class_stop_frac.tolist())),
            "class_stop_n": list(map(int, class_stop_n.tolist())),
            "class_mid": [float((edges[i] + edges[i+1]) / 2.0) for i in range(max(0, edges.size - 1))],
            "target_bins_used": int(dynamic_bins),
        })
    except Exception as e:
        logger.warning("labels: class_stop_frac compute failed (%s/%s): %s", symbol, pure_strategy, e)

    _save_label_table(
        df,
        symbol,
        pure_strategy,
        gains,
        labels,
        edges,
        bin_counts,
        bin_spans,
        extra_cols=extra_cols,
        extra_meta=extra_meta,
        group_id=group_id,
    )

    empty_bins = int(np.sum(bin_counts == 0))
    num_classes = int(edges.size - 1)
    logger.info(
        "labels: freeze %s/%s bins(request=%d,used=%d) empty=%d -> NUM_CLASSES=%d counts=%s",
        symbol,
        pure_strategy,
        int(_TARGET_BINS),
        int(dynamic_bins),
        empty_bins,
        num_classes,
        _counts_dict(labels),
    )

    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size - 1)]
    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        edges.astype(float),
        bin_counts.astype(int),
        bin_spans.astype(float),
    )

# ============================
# Public API (explicit horizon)
# ============================
def make_labels_for_horizon(
    df: pd.DataFrame,
    symbol: str,
    horizon_hours: int,
    group_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], str, np.ndarray, np.ndarray, np.ndarray]:
    # _future_extreme_signed_returns는 dn + up 형태로 준다고 가정
    n = len(df)
    both = _future_extreme_signed_returns(df, horizon_hours=int(horizon_hours))
    if both is None:
        dn = np.zeros(n, dtype=np.float32)
        up = np.zeros(n, dtype=np.float32)
    else:
        dn = np.asarray(both[:n], dtype=np.float32)
        up = np.asarray(both[n:], dtype=np.float32)

    # 1) 분포는 둘 다 넣어서
    dist_for_bins = np.concatenate([dn, up], axis=0)

    # 🔸 df 길이에 맞춰 target_bins 동적 결정
    dynamic_bins = _auto_target_bins(len(df))
    edges, bin_counts, bin_spans = _build_bins(dist_for_bins, dynamic_bins)

    # 2) 라벨은 더 크게 움직인 쪽
    gains = _pick_per_candle_gain(up, dn)

    strategy = "단기" if horizon_hours <= 4 else ("중기" if horizon_hours <= 24 else "장기")

    extra_cols = {
        "future_up": up,
        "future_dn": dn,
    }
    sl = 0.02
    extra_cols["up_ge_2pct"] = (up >= sl).astype(np.int8)
    extra_cols["dn_le_-2pct"] = (dn <= -sl).astype(np.int8)
    extra_cols["conflict_2pct"] = ((up >= sl) & (dn <= -sl)).astype(np.int8)

    labels = _vector_bin(gains, edges)

    extra_meta = {}
    try:
        class_stop_frac, class_stop_n = _compute_class_stop_frac(
            labels,
            edges,
            up,
            dn,
            stoploss_abs=sl,
        )
        extra_meta.update({
            "stoploss_threshold_abs": float(sl),
            "class_stop_frac": list(map(float, class_stop_frac.tolist())),
            "class_stop_n": list(map(int, class_stop_n.tolist())),
            "class_mid": [float((edges[i] + edges[i+1]) / 2.0) for i in range(max(0, edges.size - 1))],
            "target_bins_used": int(dynamic_bins),
        })
    except Exception as e:
        logger.warning("labels(h=%s): class_stop_frac compute failed (%s/%s): %s", horizon_hours, symbol, strategy, e)

    _save_label_table(
        df,
        symbol,
        strategy,
        gains,
        labels,
        edges,
        bin_counts,
        bin_spans,
        extra_cols=extra_cols,
        extra_meta=extra_meta,
        group_id=group_id,
    )

    empty_bins = int(np.sum(bin_counts == 0))
    num_classes = int(edges.size - 1)
    logger.info(
        "labels(h=%s): freeze %s/%s bins(request=%d,used=%d) empty=%d -> NUM_CLASSES=%d counts=%s",
        horizon_hours,
        symbol,
        strategy,
        int(_TARGET_BINS),
        int(dynamic_bins),
        empty_bins,
        num_classes,
        _counts_dict(labels),
    )

    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size - 1)]
    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        strategy,
        edges.astype(float),
        bin_counts.astype(int),
        bin_spans.astype(float),
    )

def make_all_horizon_labels(
    df: pd.DataFrame,
    symbol: str,
    horizons: List[int] | None = None,
    group_id: int | None = None,
) -> Dict[str, tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]]:
    if horizons is None:
        horizons = [4, 24, 168]

    out: Dict[str, tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]] = {}
    for h in horizons:
        gains, labels, ranges, strategy, edges, counts, spans = make_labels_for_horizon(
            df,
            symbol,
            h,
            group_id=group_id,
        )
        key = f"{h}h" if h < 24 else ("1d" if h == 24 else (f"{h//24}d" if h < 168 else "7d"))
        out[key] = (gains, labels, ranges, edges, counts, spans)
        logger.debug("make_all_horizon_labels: %s -> strategy=%s, key=%s", h, strategy, key)
    return out
