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
    BOUNDARY_BAND,  # (현재 파일에선 사용하지 않지만 호환 유지)
    _strategy_horizon_hours,
    _future_extreme_signed_returns,
    get_BIN_META,
    get_CLASS_BIN,
)

logger = logging.getLogger(__name__)

# ===== 설정: config 우선, env는 보조 =====
_BIN_META = dict(get_BIN_META() or {})

def _as_ratio(x: float) -> float:
    """0.003(비율) 또는 0.3(%) 형태가 들어와도 항상 비율[0~1]로 정규화"""
    try:
        xv = float(x)
    except Exception:
        return 0.0
    return xv / 100.0 if xv >= 1.0 else xv

def _as_percent(x: float) -> float:
    """0.08(비율) 또는 8.0(%) 형태가 들어와도 항상 '퍼센트 수치'로 정규화 (예: 8.0)"""
    try:
        xv = float(x)
    except Exception:
        return 0.0
    return xv * 100.0 if 0.0 < xv < 1.0 else xv

# ---- BIN META 읽기 (env → config 우선순위, 단위 정규화 포함) ----
_TARGET_BINS = int(os.getenv("TARGET_BINS", str(_BIN_META.get("TARGET_BINS", 8))))
_OUT_Q_LOW = float(os.getenv("OUTLIER_Q_LOW", str(_BIN_META.get("OUTLIER_Q_LOW", 0.01))))
_OUT_Q_HIGH = float(os.getenv("OUTLIER_Q_HIGH", str(_BIN_META.get("OUTLIER_Q_HIGH", 0.99))))

# 단일 bin 폭 상한(절대 %)
_MAX_BIN_SPAN_PCT = _as_percent(float(os.getenv("MAX_BIN_SPAN_PCT", str(_BIN_META.get("MAX_BIN_SPAN_PCT", 8.0)))))

# 최소 샘플 비율(과거 로직 잔존용; 본 개정본에서는 임의 병합을 하지 않으므로 사용하지 않음)
_MIN_BIN_COUNT_FRAC = float(os.getenv("MIN_BIN_COUNT_FRAC", str(_BIN_META.get("MIN_BIN_COUNT_FRAC", 0.05))))

# 추가 메타(지배적 bin 제어, 센터 밴드 상한)
_DOMINANT_MAX_FRAC = float(os.getenv("DOMINANT_MAX_FRAC", str(_BIN_META.get("DOMINANT_MAX_FRAC", 0.35))))
_DOMINANT_MAX_ITERS = int(os.getenv("DOMINANT_MAX_ITERS", str(_BIN_META.get("DOMINANT_MAX_ITERS", 6))))

# 중앙 밴드 상한(%) — config 혼재 단위 → %로 정규화
_CENTER_SPAN_MAX_PCT = _as_percent(float(os.getenv("CENTER_SPAN_MAX_PCT", str(_BIN_META.get("CENTER_SPAN_MAX_PCT", 0.5)))))

# CLASS_BIN zero-band 힌트(없으면 중앙폭 상한과 동일 취급) → %로 정규화
_CLASS_BIN_META: Dict = dict(get_CLASS_BIN() or {})
_ZERO_BAND_PCT_HINT = _as_percent(float(_CLASS_BIN_META.get("ZERO_BAND_PCT", _CENTER_SPAN_MAX_PCT)))

# === 퍼시스턴트 저장소(엣지/라벨 고정) ===
# 여기서 /persistent 못 만들면 /tmp/... 로 자동 폴백하도록 바꿨다.
def _ensure_dir_with_fallback(primary: str, fallback: str) -> Path:
    p_primary = Path(primary).resolve()
    try:
        p_primary.mkdir(parents=True, exist_ok=True)
        return p_primary
    except Exception as e:
        # 권한 같은 걸로 실패하면 /tmp 쪽으로 떨어진다.
        logger.warning("labels: can't create %s (%s) -> fallback to %s", primary, e, fallback)
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

# 아주 작은 수익률은 학습 제외(기본 ±0.3%) — (참고 상수, 본 파일의 라벨링은 전체 표본 사용)
_RAW_MIN_GAIN_FOR_TRAIN = float(os.getenv("MIN_GAIN_FOR_TRAIN", "0.003"))
_MIN_GAIN_FOR_TRAIN = _as_ratio(_RAW_MIN_GAIN_FOR_TRAIN)

# 라벨 안정화 상수
_MIN_CLASS_FRAC = 0.01
_MIN_CLASS_ABS = 8
_Q_EPS = 1e-9
_EDGE_EPS = 1e-12

# 커버리지 회복을 위한 동적 완화 한계(참고)
_RAW_MIN_GAIN_LOWER_BOUND = float(os.getenv("MIN_GAIN_LOWER_BOUND", "0.0005"))
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
    # max_span_pct는 '퍼센트 수치' (예: 8.0 = 8%) 기준
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
    """
    ✅ 임의 병합 금지 정책:
    - 카운트가 0인 bin만 제거하여 엣지를 정리한다.
    - 카운트>0 bin은 소수라도 유지한다.
    """
    if edges.size < 2 or counts.size != edges.size - 1:
        return edges.astype(float), counts.astype(int) if counts is not None else np.zeros(0, dtype=int)

    keep_mask = counts.astype(int) > 0
    if np.all(keep_mask):
        return edges.astype(float), counts.astype(int)

    # edges: [e0, e1, e2, e3, ...] / counts: [c0, c1, c2, ...]
    # c_i == 0 인 간격 [e_i, e_{i+1}] 을 삭제 → e_{i+1} 제거
    new_edges = [float(edges[0])]
    new_counts = []
    for i, cnt in enumerate(counts):
        if int(cnt) > 0:
            new_edges.append(float(edges[i+1]))
            new_counts.append(int(cnt))
        else:
            # 빈 구간은 스킵(엣지 상단을 건너뜀)
            pass

    if len(new_edges) < 2:  # 모든 구간이 비었을 때는 원본 유지(최소 형태)
        return edges.astype(float), counts.astype(int)

    return _dedupe_edges(np.array(new_edges, dtype=float)), np.array(new_counts, dtype=int)

def _enforce_zero_band(edges: np.ndarray, zero_band_pct: float) -> np.ndarray:
    # zero_band_pct는 '퍼센트 수치' (예: 0.3 = 0.3%) 기준
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
    # center_span_max_pct는 '퍼센트 수치' 기준
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
    """
    ✅ 핵심 변경점
    - 동적 등빈분할 기반으로 엣지 생성
    - 과폭 분할 보정
    - ✅ **빈 bin(카운트 0)만 제거** (임의 병합 금지)
    - 지배적 bin 분할 + 중앙폭 제한
    - zero-band 보정 + 0 경계 보장
    """
    x = np.asarray(gains, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        edges = np.array([-0.05, -0.02, -0.005, 0.005, 0.02, 0.05], dtype=float)
        counts = np.zeros(edges.size - 1, dtype=int)
        spans = np.diff(edges) * 100.0
        return edges, counts, spans

    # 1) 이상치 클리핑
    x_clip, _, _ = _clip_outliers(x)

    # 2) 초기 등빈 엣지
    k = max(2, int(target_bins))
    edges = _equal_freq_edges(x_clip, k)

    # 3) 과폭 분할 보정
    edges = _split_wide_bins(edges, _MAX_BIN_SPAN_PCT)

    # 4) 현재 카운트 측정
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    counts, _ = np.histogram(x_clip, bins=edges_count)

    # 5) ✅ 빈 bin만 제거 (임의 병합 금지)
    edges, counts = _drop_empty_bins(edges, counts)

    # 6) 지배적 bin 분할 + 중앙 폭 제한
    edges = _limit_dominant_bins(
        edges, x_clip,
        max_frac=float(_DOMINANT_MAX_FRAC),
        max_iters=int(_DOMINANT_MAX_ITERS),
        center_span_max_pct=float(_CENTER_SPAN_MAX_PCT),
    )

    # 7) zero-band 보정 + 0 경계 보장
    edges = _enforce_zero_band(edges, _ZERO_BAND_PCT_HINT)
    edges = _ensure_zero_edge(edges)

    # 8) 최종 카운트/스팬 재계산
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    counts, _ = np.histogram(x_clip, bins=edges_count)
    spans_pct = np.diff(edges) * 100.0

    return edges.astype(float), counts.astype(int), spans_pct.astype(float)

# ============================
# Stable Edge Store I/O
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
    extra_meta: Dict | None = None,
) -> None:
    """라벨 테이블과 메타를 파일로 고정 저장 (Parquet 우선, 실패 시 CSV 폴백)
       extra_cols: 보조 컬럼(예: 손절 경유 플래그들)
       extra_meta: 경계 JSON에 추가로 함께 저장할 메타(예: class_stop_frac 등)"""
    ts = _to_series_ts_kst(df["timestamp"]) if "timestamp" in df.columns else pd.Series(pd.NaT, index=range(len(gains)))
    out = pd.DataFrame({
        "ts": ts,
        "symbol": symbol,
        "strategy": strategy,
        "class_id": labels.astype(np.int64),
        "signed_gain": gains.astype(np.float32),
    })
    if extra_cols:
        for k, v in extra_cols.items():
            try:
                out[k] = np.asarray(v)
            except Exception:
                # 길이 불일치나 타입 오류는 경고만 남기고 무시
                logger.warning("labels: failed to attach extra column '%s'", k)

    p_parquet = _labels_path(symbol, strategy)
    p_csv = _labels_csv_path(symbol, strategy)
    p_parquet.parent.mkdir(parents=True, exist_ok=True)

    # 1) Parquet 저장 시도
    try:
        out.to_parquet(p_parquet, index=False)
        logger.info("labels: table saved -> %s (%s/%s) rows=%d", str(p_parquet), symbol, strategy, len(out))
    except Exception as e:
        # 2) CSV 폴백
        try:
            out.to_csv(p_csv, index=False)
            logger.warning(
                "labels: failed to save label table (%s/%s) to Parquet (%s): %s -> fallback to CSV (%s) rows=%d",
                symbol, strategy, str(p_parquet), e, str(p_csv), len(out)
            )
        except Exception as e2:
            logger.warning(
                "labels: failed to save label table (%s/%s) to CSV (%s) as well: %s",
                symbol, strategy, str(p_csv), e2
            )

    # 메타 로그(요약) + 엣지 JSON 저장
    num_classes = int(max(0, edges.size - 1))
    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(num_classes)]
    meta = {
        "symbol": symbol,
        "strategy": strategy,
        "NUM_CLASSES": num_classes,
        "class_counts_label_freeze": _counts_dict(labels),
        "edges": list(map(float, edges.tolist())),
        "edges_hash": _hash_array(edges),
        "bin_counts": list(map(int, counts.tolist())) if counts is not None else [],
        "bin_spans_pct": list(map(float, spans_pct.tolist())) if spans_pct is not None else [],
        # ✅ 여기 추가된 부분: trainer가 줄이지 말라는 힌트
        "dynamic_classes": True,
        "allow_trainer_class_collapse": False,
        "class_ranges": class_ranges,
    }
    if isinstance(extra_meta, dict) and extra_meta:
        try:
            meta.update(extra_meta)
        except Exception:
            pass
    _save_edges(symbol, strategy, edges, meta=meta)  # 경계 JSON에도 메타 함께 반영

# ============================
# 내부 유틸: 클래스별 손절 경유 비율 집계
# ============================
def _compute_class_stop_frac(
    labels: np.ndarray,
    edges: np.ndarray,
    up: np.ndarray,
    dn: np.ndarray,
    stoploss_abs: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 클래스 c에 대해:
      - mid = (lo+hi)/2
      - mid>0 (롱 성향): 손절경유 = (dn <= -SL)
      - mid<0 (숏 성향): 손절경유 = (up >=  SL)
      - mid≈0 (중립): 손절경유 = (dn <= -SL) OR (up >= SL)
    반환:
      class_stop_frac: (C,) 각 클래스 내 손절경유 비율
      class_stop_n:    (C,) 각 클래스 내 표본 수
    """
    C = max(0, edges.size - 1)
    if C <= 0 or labels.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    class_stop_frac = np.zeros(C, dtype=float)
    class_stop_n = np.zeros(C, dtype=int)

    # 클래스 미드사인 미리 계산
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
        if mid > 0.0:          # 롱 성향 클래스
            stop_hit = (dn[idx] <= -abs(stoploss_abs))
        elif mid < 0.0:        # 숏 성향 클래스
            stop_hit = (up[idx] >=  abs(stoploss_abs))
        else:                   # 중립 클래스
            stop_hit = (dn[idx] <= -abs(stoploss_abs)) | (up[idx] >= abs(stoploss_abs))
        class_stop_frac[c] = float(np.mean(stop_hit.astype(np.float32)))

    return class_stop_frac, class_stop_n

# ============================
# Public API (strategy-based)
# ============================
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
    # 1) 수익률 계산(분포용): '가장 크게 움직인 방향'의 서명 수익률
    gains = signed_future_return(df, strategy)  # (N,)

    # 1-추가) 손절 경유 보조 플래그(±2%) — 라벨엔 반영하지 않음, 테이블에만 저장
    extra_cols = None
    up = dn = None
    sl = 0.02
    try:
        horizon_hours = _strategy_horizon_hours(strategy)
        both = _future_extreme_signed_returns(df, horizon_hours=int(horizon_hours))
        n = len(df)
        dn = both[:n] if both is not None and both.size >= 2 * n else np.zeros(n, dtype=np.float32)
        up = both[n:] if both is not None and both.size >= 2 * n else np.zeros(n, dtype=np.float32)
        up_ge_2pct = (np.asarray(up) >= sl)
        dn_le_m2pct = (np.asarray(dn) <= -sl)
        conflict_2pct = up_ge_2pct & dn_le_m2pct
        extra_cols = {
            "up_ge_2pct": up_ge_2pct.astype(np.int8),
            "dn_le_-2pct": dn_le_m2pct.astype(np.int8),
            "conflict_2pct": conflict_2pct.astype(np.int8),
        }
    except Exception as e:
        logger.warning("labels: extra risk flags failed (%s/%s): %s", symbol, strategy, e)
        extra_cols = None

    # 2) 현재 표본으로 엣지 생성(라벨 고정용) → 저장
    edges, counts0, spans0 = _build_bins(gains, _TARGET_BINS)

    # 3) 전 표본 라벨링(경계 마스킹 없음, -1 없음)
    labels = _vector_bin(gains, edges)

    # 3-추가) 클래스별 손절 경유 비율 집계 → 메타로 저장
    extra_meta = {}
    try:
        if up is not None and dn is not None:
            class_stop_frac, class_stop_n = _compute_class_stop_frac(labels, edges, up, dn, stoploss_abs=sl)
            extra_meta.update({
                "stoploss_threshold_abs": float(sl),
                "class_stop_frac": list(map(float, class_stop_frac.tolist())),
                "class_stop_n": list(map(int, class_stop_n.tolist())),
                "class_mid": [float((edges[i]+edges[i+1])/2.0) for i in range(max(0, edges.size-1))],
            })
    except Exception as e:
        logger.warning("labels: class_stop_frac compute failed (%s/%s): %s", symbol, strategy, e)

    # 4) class_ranges / counts / spans
    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size - 1)]
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    bin_counts, _ = np.histogram(np.clip(gains, edges[0], edges[-1]), bins=edges_count)
    bin_spans = np.diff(edges) * 100.0

    # 5) 라벨/엣지 고정 저장(Parquet 우선 저장, 실패 시 CSV 폴백) + 경계 JSON(+extra_meta)
    _save_label_table(df, symbol, strategy, gains, labels, edges, bin_counts, bin_spans, extra_cols=extra_cols, extra_meta=extra_meta)

    # 6) 로그 요약 (체크리스트 스타일)
    empty_bins = int(np.sum(bin_counts == 0))
    num_classes = int(edges.size - 1)
    logger.info(
        "labels: freeze %s/%s bins(fd)=%d, empty=%d -> NUM_CLASSES=%d counts=%s",
        symbol, strategy, int(_TARGET_BINS), empty_bins, num_classes, _counts_dict(labels)
    )

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
    # 1) 수익률 & 전략 매핑
    gains = signed_future_return_by_hours(df, horizon_hours=int(horizon_hours))
    strategy = _strategy_from_hours(int(horizon_hours))

    # 1-추가) 손절 경유 보조 플래그(±2%) — 라벨엔 반영하지 않음
    extra_cols = None
    up = dn = None
    sl = 0.02
    try:
        both = _future_extreme_signed_returns(df, horizon_hours=int(horizon_hours))
        n = len(df)
        dn = both[:n] if both is not None and both.size >= 2 * n else np.zeros(n, dtype=np.float32)
        up = both[n:] if both is not None and both.size >= 2 * n else np.zeros(n, dtype=np.float32)
        up_ge_2pct = (np.asarray(up) >= sl)
        dn_le_m2pct = (np.asarray(dn) <= -sl)
        conflict_2pct = up_ge_2pct & dn_le_m2pct
        extra_cols = {
            "up_ge_2pct": up_ge_2pct.astype(np.int8),
            "dn_le_-2pct": dn_le_m2pct.astype(np.int8),
            "conflict_2pct": conflict_2pct.astype(np.int8),
        }
    except Exception as e:
        logger.warning("labels(h=%s): extra risk flags failed (%s/%s): %s", horizon_hours, symbol, strategy, e)
        extra_cols = None

    # 2) 현재 표본으로 엣지 생성(라벨 고정용) → 저장
    edges, counts0, spans0 = _build_bins(gains, _TARGET_BINS)

    # 3) 전 표본 라벨링
    labels = _vector_bin(gains, edges)

    # 3-추가) 클래스별 손절 경유 비율 집계 → 메타로 저장
    extra_meta = {}
    try:
        if up is not None and dn is not None:
            class_stop_frac, class_stop_n = _compute_class_stop_frac(labels, edges, up, dn, stoploss_abs=sl)
            extra_meta.update({
                "stoploss_threshold_abs": float(sl),
                "class_stop_frac": list(map(float, class_stop_frac.tolist())),
                "class_stop_n": list(map(int, class_stop_n.tolist())),
                "class_mid": [float((edges[i]+edges[i+1])/2.0) for i in range(max(0, edges.size-1))],
            })
    except Exception as e:
        logger.warning("labels(h=%s): class_stop_frac compute failed (%s/%s): %s", horizon_hours, symbol, strategy, e)

    # 4) class_ranges / counts / spans
    class_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(edges.size - 1)]
    edges_count = edges.copy(); edges_count[-1] += _EDGE_EPS
    bin_counts, _ = np.histogram(np.clip(gains, edges[0], edges[-1]), bins=edges_count)
    bin_spans = np.diff(edges) * 100.0

    # 5) 저장 (Parquet 우선, 실패 시 CSV 폴백) + 경계 JSON(+extra_meta)
    _save_label_table(df, symbol, strategy, gains, labels, edges, bin_counts, bin_spans, extra_cols=extra_cols, extra_meta=extra_meta)

    # 6) 로그 요약 (체크리스트 스타일)
    empty_bins = int(np.sum(bin_counts == 0))
    num_classes = int(edges.size - 1)
    logger.info(
        "labels(h=%s): freeze %s/%s bins(fd)=%d, empty=%d -> NUM_CLASSES=%d counts=%s",
        horizon_hours, symbol, strategy, int(_TARGET_BINS), empty_bins, num_classes, _counts_dict(labels)
    )

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
