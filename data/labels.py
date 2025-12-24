# ================================================
# labels.py — YOPO RAW 기반 수익률 라벨링
#            (H 복구 + 동적 엣지 튜닝 + 희소 클래스 병합 옵션 버전)
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
    # ✅ config.py의 persistent root를 그대로 사용 (핵심)
    PERSISTENT_DIR,
    _future_extreme_signed_returns,
    get_BIN_META,
    get_SPARSE_CLASS,
    TRAIN_ZERO_BAND_ABS,
    get_CLASS_BIN,   # ✅ 추가: 1% 노트레이드 컷 가져오기
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

# 최소 패턴 수
_MIN_SAMPLES_PER_CLASS = int(os.getenv("MIN_SAMPLES_PER_CLASS", "50"))

# 희소 클래스 병합 옵션
_SPARSE_CLASS_CONF = dict(get_SPARSE_CLASS() or {})
_SC_MIN_SAMPLES = int(_SPARSE_CLASS_CONF.get("MIN_SAMPLES_PER_CLASS", 12))
_SC_MIN_CLASSES = int(_SPARSE_CLASS_CONF.get("MIN_CLASSES_AFTER_MERGE", 8))
_SC_MAX_PASSES = int(_SPARSE_CLASS_CONF.get("MAX_MERGE_PASSES", 2))

MERGE_SPARSE_LABEL_BINS = os.getenv("MERGE_SPARSE_LABEL_BINS", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# 극단 꼬리 trim
_TAIL_TRIM_FRAC = float(os.getenv("LABEL_TAIL_TRIM_FRAC", "0.005"))

# ============================================================
# ✅ (핵심 수정) "캔들 1개 내부 high/low" 기준으로 통일
# ------------------------------------------------------------
# - 단기/중기/장기 모두 H=1
# - 미래 여러 캔들 극단값 금지
# ============================================================
_FIXED_H = {
    "단기": 1,
    "중기": 1,
    "장기": 1,
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

# ============================================================
# ✅ 저장 경로 통일 (config.py의 PERSISTENT_DIR 기준)
# ============================================================
_PERSIST_BASE = str(PERSISTENT_DIR)

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

# ============================================================
# 미래 구간 H 반환 — (✅ 통일: 항상 1캔들)
# ============================================================
def _get_fixed_horizon_candles(strategy: str) -> int:
    pure = _normalize_strategy_name(strategy)
    return int(_FIXED_H.get(pure, 1))

# ============================================================
# ✅ 수익률 계산 (캔들 1개 내부 high/low)
# ============================================================
def _future_extreme_signed_returns_by_candles(df: pd.DataFrame, H: int):
    """
    각 시작 캔들 i 에 대해:
    - base = 해당 시점 close
    - ✅ 현재 캔들 1개에서 high/low 사용
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    high  = pd.to_numeric(df.get("high", df["close"]), errors="coerce").to_numpy(dtype=np.float64)
    low   = pd.to_numeric(df.get("low",  df["close"]), errors="coerce").to_numpy(dtype=np.float64)

    up = np.zeros(n, dtype=np.float32)
    dn = np.zeros(n, dtype=np.float32)

    # ✅ 무조건 1캔들
    H = 1

    for i in range(n):
        base = close[i]
        if not np.isfinite(base) or base <= 0:
            continue

        j = min(n, i + H)
        window_high = high[i:j]
        window_low  = low[i:j]

        valid_high = window_high[np.isfinite(window_high)]
        valid_low  = window_low[np.isfinite(window_low)]

        if valid_high.size == 0 or valid_low.size == 0:
            continue

        future_high = float(valid_high.max())
        future_low  = float(valid_low.min())

        if not np.isfinite(future_high) or not np.isfinite(future_low):
            continue

        base_safe = base + 1e-12
        up[i] = float((future_high - base) / base_safe)
        dn[i] = float((future_low  - base) / base_safe)

    return up.astype(np.float32), dn.astype(np.float32)

# ============================================================
# RAW gain 선택
# ============================================================
def _pick_per_candle_gain(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    return np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)

# ============================================================
# RAW bin 계산 (동적 quantile + trim)
# ============================================================
def _raw_bins(dist: np.ndarray, target_bins: int) -> np.ndarray:
    if dist is None:
        return np.linspace(-0.01, 0.01, target_bins + 1).astype(float)

    dist = np.asarray(dist, dtype=float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return np.linspace(-0.01, 0.01, target_bins + 1).astype(float)

    n = dist.size

    if _TAIL_TRIM_FRAC > 0.0 and n >= 100:
        try:
            q_low, q_high = np.quantile(dist, [_TAIL_TRIM_FRAC, 1.0 - _TAIL_TRIM_FRAC])
            mask = (dist >= q_low) & (dist <= q_high)
            trimmed = dist[mask]
            if trimmed.size >= max(_MIN_LABEL_CLASSES * 2, int(0.7 * n)):
                dist = trimmed
                n = dist.size
        except Exception:
            pass

    lo = float(np.min(dist))
    hi = float(np.max(dist))
    if not np.isfinite(lo):
        lo = -0.01
    if not np.isfinite(hi):
        hi = 0.01
    if hi <= lo:
        hi = lo + 1e-6

    if n < _MIN_SAMPLES_PER_CLASS * 2:
        return np.linspace(lo, hi, max(_MIN_LABEL_CLASSES, min(target_bins, 4)) + 1).astype(float)

    max_bins_by_samples = max(1, n // max(1, _MIN_SAMPLES_PER_CLASS))
    bins = int(min(target_bins, max_bins_by_samples))
    bins = int(max(_MIN_LABEL_CLASSES, bins))

    if bins <= 1:
        return np.linspace(lo, hi, 3).astype(float)

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
# ✅ (핵심 추가) 숏/롱 분리 bin 생성 + 1% 미만 클래스 생성 금지
# ============================================================
def _get_no_trade_floor_abs() -> float:
    """
    YOPO 규칙: abs(return) < 1% 는 예측도/학습도 의미없음.
    config.CLASS_BIN.no_trade_floor_abs (기본 0.01)를 사용.
    """
    try:
        cb = dict(get_CLASS_BIN() or {})
        return float(cb.get("no_trade_floor_abs", 0.01))
    except Exception:
        return 0.01

def _split_edges_short_long(gains: np.ndarray, floor: float, target_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    - 숏: gains <= -floor
    - 롱: gains >= +floor
    각각 따로 quantile edges 생성.
    """
    g = np.asarray(gains, dtype=float)
    g = g[np.isfinite(g)]
    neg = g[g <= -floor]
    pos = g[g >=  floor]

    if neg.size < max(50, _MIN_SAMPLES_PER_CLASS * 2):
        neg = None
    if pos.size < max(50, _MIN_SAMPLES_PER_CLASS * 2):
        pos = None

    if neg is None and pos is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    # ✅ 한쪽만 있을 때도 floor 클램프 후 단조 증가 유지 보정
    if neg is None:
        bins_pos = max(_MIN_LABEL_CLASSES, target_bins)
        edges_pos = _raw_bins(pos, bins_pos)

        edges_pos = np.asarray(edges_pos, dtype=float)
        if edges_pos.size >= 2:
            edges_pos[0] = max(float(edges_pos[0]), float(floor))
            for i in range(1, edges_pos.size):
                if edges_pos[i] <= edges_pos[i - 1]:
                    edges_pos[i] = edges_pos[i - 1] + 1e-9
        return np.array([], dtype=float), edges_pos

    if pos is None:
        bins_neg = max(_MIN_LABEL_CLASSES, target_bins)
        edges_neg = _raw_bins(neg, bins_neg)

        edges_neg = np.asarray(edges_neg, dtype=float)
        if edges_neg.size >= 2:
            edges_neg[-1] = min(float(edges_neg[-1]), float(-floor))
            for i in range(1, edges_neg.size):
                if edges_neg[i] <= edges_neg[i - 1]:
                    edges_neg[i] = edges_neg[i - 1] + 1e-9
        return edges_neg, np.array([], dtype=float)

    total = neg.size + pos.size
    frac_neg = neg.size / max(1, total)
    bins_neg = int(round(target_bins * frac_neg))
    bins_pos = target_bins - bins_neg
    bins_neg = max(_MIN_LABEL_CLASSES, min(target_bins - _MIN_LABEL_CLASSES, bins_neg))
    bins_pos = max(_MIN_LABEL_CLASSES, target_bins - bins_neg)

    edges_neg = _raw_bins(neg, bins_neg)
    edges_pos = _raw_bins(pos, bins_pos)

    # 끝/시작을 floor에 맞춰 깔끔하게
    edges_neg[-1] = -floor
    edges_pos[0]  =  floor

    # 단조 증가 보정
    for i in range(1, edges_neg.size):
        if edges_neg[i] <= edges_neg[i - 1]:
            edges_neg[i] = edges_neg[i - 1] + 1e-9
    for i in range(1, edges_pos.size):
        if edges_pos[i] <= edges_pos[i - 1]:
            edges_pos[i] = edges_pos[i - 1] + 1e-9

    return edges_neg.astype(float), edges_pos.astype(float)

def _bin_split_short_long(gains: np.ndarray, edges_neg: np.ndarray, edges_pos: np.ndarray, floor: float) -> np.ndarray:
    """
    숏/롱 각각 binning 후,
    - 숏 라벨: 0..(nneg-1)
    - 롱 라벨: nneg..(nneg+npos-1)
    - abs(return)<floor 는 -1
    """
    g = np.asarray(gains, dtype=float)
    labels = np.full(g.shape[0], -1, dtype=np.int64)

    nneg = int(max(0, edges_neg.size - 1))
    npos = int(max(0, edges_pos.size - 1))

    if nneg > 0:
        mneg = np.isfinite(g) & (g <= -floor)
        if np.any(mneg):
            labels[mneg] = _vector_bin(g[mneg].astype(np.float32), edges_neg.astype(float))

    if npos > 0:
        mpos = np.isfinite(g) & (g >= floor)
        if np.any(mpos):
            labels[mpos] = _vector_bin(g[mpos].astype(np.float32), edges_pos.astype(float)) + nneg

    return labels

# ============================================================
# 희소 bin 병합 (옵션)
# ============================================================
def _merge_sparse_bins(edges: np.ndarray, values: np.ndarray):
    try:
        edges = np.asarray(edges, dtype=float)
        if edges.size < 3:
            return edges, np.zeros(max(0, edges.size - 1), dtype=int)

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return edges, np.zeros(max(0, edges.size - 1), dtype=int)

        e2 = edges.copy()
        e2[-1] += 1e-12
        counts, _ = np.histogram(values, bins=e2)
        counts = counts.astype(int)

        total = int(counts.sum())
        if total <= 0:
            return edges, counts

        min_samples = max(1, _SC_MIN_SAMPLES)
        min_classes = max(1, _SC_MIN_CLASSES)
        max_passes = max(0, _SC_MAX_PASSES)

        if counts.size <= min_classes or max_passes == 0:
            return edges, counts

        e = edges.copy()
        c = counts.copy()

        for _ in range(max_passes):
            if c.size <= min_classes:
                break

            sparse_idx = np.where(c < min_samples)[0]
            if sparse_idx.size == 0:
                break

            sparse_idx = list(sorted(sparse_idx, key=lambda i: c[i]))
            changed = False

            for idx in sparse_idx:
                if c.size <= min_classes:
                    break
                if idx >= c.size:
                    continue
                if c[idx] >= min_samples:
                    continue

                left_ok = idx - 1 >= 0
                right_ok = idx + 1 < c.size
                if not (left_ok or right_ok):
                    continue

                if left_ok and right_ok:
                    if c[idx - 1] >= c[idx + 1]:
                        nbr = idx - 1
                    else:
                        nbr = idx + 1
                elif left_ok:
                    nbr = idx - 1
                else:
                    nbr = idx + 1

                if nbr == idx - 1:
                    c[nbr] += c[idx]
                    c = np.delete(c, idx)
                    e = np.delete(e, idx)
                elif nbr == idx + 1:
                    c[idx] += c[nbr]
                    c = np.delete(c, nbr)
                    e = np.delete(e, nbr)

                changed = True

            if not changed:
                break

        if e.size != c.size + 1:
            logger.warning("merge_sparse_bins: edge/count mismatch → 원본 유지")
            return edges, counts

        return e, c

    except Exception as ex:
        logger.warning("merge_sparse_bins failed: %s", ex)
        e2 = edges.copy()
        e2[-1] += 1e-12
        counts, _ = np.histogram(values, bins=e2)
        return edges, counts.astype(int)

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
# 수익률 계산
# ============================================================
def compute_label_returns(df: pd.DataFrame, symbol: str, strategy: str):
    pure = _normalize_strategy_name(strategy)
    H = _get_fixed_horizon_candles(pure)  # ✅ 항상 1
    up, dn = _future_extreme_signed_returns_by_candles(df, H)
    gains = _pick_per_candle_gain(up, dn)
    target = _auto_target_bins(len(df))
    return gains, up, dn, target

# ============================================================
# 저장 헬퍼
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
        "edges": list(map(float, edges.tolist())) if isinstance(edges, np.ndarray) else edges,
        "edges_hash": _hash_array(edges) if isinstance(edges, np.ndarray) else "na",
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
# 라벨 테이블 저장
# ============================================================
def _save_label_table(df, symbol, strategy, gains, labels, class_ranges, counts,
                      spans, extra_cols=None, extra_meta=None, group_id=None,
                      edges_meta=None):

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

    meta = {
        "symbol": symbol,
        "strategy": pure,
        "NUM_CLASSES": int(len(class_ranges)),
        "class_ranges": class_ranges,
        "bin_counts": list(map(int, counts)),
    }
    if extra_meta:
        meta.update(extra_meta)
    if group_id is not None:
        meta["group_id"] = group_id

    # edges는 메타용(보기/디버그용)으로만 저장
    if edges_meta is None:
        edges_meta = []
    _save_edges(symbol, pure, edges_meta, meta)

# ============================================================
# make_labels
# ============================================================
def make_labels(df, symbol, strategy, group_id=None):
    pure = _normalize_strategy_name(strategy)

    gains, up_c, dn_c, target_bins = compute_label_returns(df, symbol, pure)

    # ✅ YOPO 규칙: 1% 미만(숏/롱 모두) 학습/예측 의미없음 → 학습에서 제거
    floor_abs = float(os.getenv("LABEL_NO_TRADE_FLOOR_ABS", str(_get_no_trade_floor_abs())))

    # ✅ 학습 제외 밴드: 0% 근처 제외 + 1% 노트레이드 컷 중 더 큰 값 적용
    hard_min_abs = float(max(float(TRAIN_ZERO_BAND_ABS), float(floor_abs)))

    train_mask = (np.isfinite(gains) & (np.abs(gains) >= hard_min_abs))

    # ✅ 숏/롱 분리해서 edges 생성 (중앙(±1%) 클래스 자체를 생성하지 않음)
    edges_neg, edges_pos = _split_edges_short_long(gains, floor_abs, target_bins)

    # 희소 병합 옵션: 숏/롱 각각 따로 적용 (학습샘플 기준)
    if MERGE_SPARSE_LABEL_BINS:
        if edges_neg.size >= 3:
            edges_neg, _ = _merge_sparse_bins(edges_neg, gains[train_mask & (gains <= -floor_abs)])
        if edges_pos.size >= 3:
            edges_pos, _ = _merge_sparse_bins(edges_pos, gains[train_mask & (gains >=  floor_abs)])

    labels = _bin_split_short_long(gains, edges_neg, edges_pos, floor_abs).astype(np.int64)

    # ✅ 학습 제외 샘플은 -1로 명확히 표시 (0 근처/1% 미만 모두 포함)
    labels[~train_mask] = -1

    # class_ranges 생성 (중앙 구간 없음)
    class_ranges: List[Tuple[float, float]] = []
    if edges_neg.size >= 2:
        for i in range(edges_neg.size - 1):
            class_ranges.append((float(edges_neg[i]), float(edges_neg[i + 1])))
    if edges_pos.size >= 2:
        for i in range(edges_pos.size - 1):
            class_ranges.append((float(edges_pos[i]), float(edges_pos[i + 1])))

    # bin_counts도 학습 샘플 기준
    counts = np.zeros(len(class_ranges), dtype=int)
    if len(class_ranges) > 0:
        # 숏 counts
        if edges_neg.size >= 2:
            e2 = edges_neg.copy()
            e2[-1] += 1e-12
            cneg, _ = np.histogram(gains[train_mask & (gains <= -floor_abs)], bins=e2)
            counts[:len(cneg)] = cneg.astype(int)
        # 롱 counts
        if edges_pos.size >= 2:
            e2 = edges_pos.copy()
            e2[-1] += 1e-12
            cpos, _ = np.histogram(gains[train_mask & (gains >=  floor_abs)], bins=e2)
            start = (edges_neg.size - 1) if edges_neg.size >= 2 else 0
            counts[start:start + len(cpos)] = cpos.astype(int)

    # spans(%)도 class_ranges 기반
    spans = np.array([(hi - lo) * 100.0 for (lo, hi) in class_ranges], dtype=float)

    sl = 0.02
    extra_cols = {
        "future_up": up_c,
        "future_dn": dn_c,
        "up_ge_2pct": (up_c >= sl).astype(np.int8),
        "dn_le_-2pct": (dn_c <= -sl).astype(np.int8),
        "train_mask": train_mask.astype(np.int8),
        "no_trade_floor_abs": np.full(len(df), float(floor_abs), dtype=np.float32),
    }

    # edges_meta는 보기용으로 저장 (숏/롱 분리 형태를 그대로 남김)
    edges_meta = {
        "floor_abs": float(floor_abs),
        "hard_min_abs": float(hard_min_abs),
        "edges_short": list(map(float, edges_neg.tolist())) if isinstance(edges_neg, np.ndarray) else [],
        "edges_long":  list(map(float, edges_pos.tolist())) if isinstance(edges_pos, np.ndarray) else [],
    }

    _save_label_table(
        df, symbol, pure,
        gains, labels,
        class_ranges,
        counts, spans,
        extra_cols=extra_cols,
        extra_meta={"target_bins_used": target_bins},
        group_id=group_id,
        edges_meta=edges_meta,
    )

    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        edges_meta,                 # ✅ 이제 edges 대신 meta(dict)를 반환(로그/저장용)
        counts.astype(int),
        spans.astype(float),
    )

# ============================================================
# make_labels_for_horizon (RAW용)  — (✅ 1캔들 통일 적용)
# ============================================================
def make_labels_for_horizon(df, symbol, horizon_hours, group_id=None):
    n = len(df)

    # ✅ 기존 _future_extreme_signed_returns(여러 캔들 극단) 대신
    # ✅ 1캔들(high/low) 방식으로 통일
    up, dn = _future_extreme_signed_returns_by_candles(df, H=1)

    target_bins = _auto_target_bins(len(df))
    gains = _pick_per_candle_gain(up, dn)

    # ✅ horizon 라벨도 동일하게 1% 컷 강제 적용
    floor_abs = float(os.getenv("LABEL_NO_TRADE_FLOOR_ABS", str(_get_no_trade_floor_abs())))
    hard_min_abs = float(max(float(TRAIN_ZERO_BAND_ABS), float(floor_abs)))
    train_mask = (np.isfinite(gains) & (np.abs(gains) >= hard_min_abs))

    edges_neg, edges_pos = _split_edges_short_long(gains, floor_abs, target_bins)

    if MERGE_SPARSE_LABEL_BINS:
        if edges_neg.size >= 3:
            edges_neg, _ = _merge_sparse_bins(edges_neg, gains[train_mask & (gains <= -floor_abs)])
        if edges_pos.size >= 3:
            edges_pos, _ = _merge_sparse_bins(edges_pos, gains[train_mask & (gains >=  floor_abs)])

    labels = _bin_split_short_long(gains, edges_neg, edges_pos, floor_abs).astype(np.int64)
    labels[~train_mask] = -1

    class_ranges: List[Tuple[float, float]] = []
    if edges_neg.size >= 2:
        for i in range(edges_neg.size - 1):
            class_ranges.append((float(edges_neg[i]), float(edges_neg[i + 1])))
    if edges_pos.size >= 2:
        for i in range(edges_pos.size - 1):
            class_ranges.append((float(edges_pos[i]), float(edges_pos[i + 1])))

    counts = np.zeros(len(class_ranges), dtype=int)
    if len(class_ranges) > 0:
        if edges_neg.size >= 2:
            e2 = edges_neg.copy()
            e2[-1] += 1e-12
            cneg, _ = np.histogram(gains[train_mask & (gains <= -floor_abs)], bins=e2)
            counts[:len(cneg)] = cneg.astype(int)
        if edges_pos.size >= 2:
            e2 = edges_pos.copy()
            e2[-1] += 1e-12
            cpos, _ = np.histogram(gains[train_mask & (gains >=  floor_abs)], bins=e2)
            start = (edges_neg.size - 1) if edges_neg.size >= 2 else 0
            counts[start:start + len(cpos)] = cpos.astype(int)

    spans = np.array([(hi - lo) * 100.0 for (lo, hi) in class_ranges], dtype=float)

    strategy = "단기" if horizon_hours <= 4 else ("중기" if horizon_hours <= 24 else "장기")

    extra_cols = {
        "future_up": up,
        "future_dn": dn,
        "up_ge_2pct": (up >= 0.02).astype(np.int8),
        "dn_le_-2pct": (dn <= -0.02).astype(np.int8),
        "train_mask": train_mask.astype(np.int8),
        "no_trade_floor_abs": np.full(len(df), float(floor_abs), dtype=np.float32),
    }

    edges_meta = {
        "floor_abs": float(floor_abs),
        "hard_min_abs": float(hard_min_abs),
        "edges_short": list(map(float, edges_neg.tolist())) if isinstance(edges_neg, np.ndarray) else [],
        "edges_long":  list(map(float, edges_pos.tolist())) if isinstance(edges_pos, np.ndarray) else [],
    }

    _save_label_table(
        df, symbol, strategy, gains, labels,
        class_ranges,
        counts, spans,
        extra_cols=extra_cols,
        extra_meta={"target_bins_used": target_bins},
        group_id=group_id,
        edges_meta=edges_meta,
    )

    return (
        gains.astype(np.float32),
        labels.astype(np.int64),
        class_ranges,
        strategy,
        edges_meta,
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
        gains, labels, ranges, strat, edges_meta, counts, spans = \
            make_labels_for_horizon(df, symbol, h, group_id)
        key = (
            f"{h}h" if h < 24 else
            ("1d" if h == 24 else
             (f"{h//24}d" if h < 168 else "7d"))
        )
        out[key] = (gains, labels, ranges, edges_meta, counts, spans)

    return out
