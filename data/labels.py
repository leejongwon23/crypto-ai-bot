# ================================================
# labels.py — YOPO RAW 기반 수익률 라벨링
#            (H 고정 + 동적 엣지 튜닝 + 희소 클래스 병합 옵션 버전)
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
    get_SPARSE_CLASS,  # 🔥 아이디어 A 설정
    TRAIN_ZERO_BAND_ABS,  # 🔥 학습용 0% 근처 절대 수익률 밴드
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

# ✅ 한 클래스당 최소 패턴 수(샘플 수) 기준
#   - 데이터가 충분하면 각 클래스에 최소 이 정도는 들어가도록 엣지를 자름
_MIN_SAMPLES_PER_CLASS = int(os.getenv("MIN_SAMPLES_PER_CLASS", "50"))

# 🔥 SPARSE_CLASS(아이디어 A) 설정: 희소 클래스 병합용
_SPARSE_CLASS_CONF = dict(get_SPARSE_CLASS() or {})
_SC_MIN_SAMPLES = int(_SPARSE_CLASS_CONF.get("MIN_SAMPLES_PER_CLASS", 12))
_SC_MIN_CLASSES = int(_SPARSE_CLASS_CONF.get("MIN_CLASSES_AFTER_MERGE", 8))
_SC_MAX_PASSES = int(_SPARSE_CLASS_CONF.get("MAX_MERGE_PASSES", 2))

# 🔥 희소 bin 병합 사용 여부 (기본 OFF)
#   - 0 (기본): bin 병합 하지 않음 → 클래스 개수 그대로 유지
#   - 1       : 아이디어 A 병합 켬  → 실험용
MERGE_SPARSE_LABEL_BINS = os.getenv("MERGE_SPARSE_LABEL_BINS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# 🔥 극단 꼬리(trim) 비율 (분포 엣지 계산용)
# - 예: 0.005 → 하위 0.5%, 상위 0.5% 정도만 "구간 계산"에서 잠깐 제외
_TAIL_TRIM_FRAC = float(os.getenv("LABEL_TAIL_TRIM_FRAC", "0.005"))

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
    """
    각 시점 i에서,
    - 진입 기준: close[i]
    - 미래 구간: i 이후 H개 캔들 (i+1 ~ i+H)
      → 현재 캔들(high/low)은 '미래'에서 제외 (근본 버그 수정 포인트)
    """
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
        # 🔥 i 이후 H개 캔들만 사용 (i+1 ~ i+H)
        start = i + 1
        end = min(n, i + 1 + H)  # 파이썬 슬라이스 end는 exclusive

        # 미래 캔들이 하나도 없으면 0으로 처리
        if start >= end:
            up[i] = 0.0
            dn[i] = 0.0
            continue

        base = close[i] if close[i] > 0 else 1e-6

        future_high = float(np.max(high[start:end]))
        future_low  = float(np.min(low[start:end]))

        up[i] = (future_high - base) / (base + 1e-12)
        dn[i] = (future_low  - base) / (base + 1e-12)

    return up, dn

# ============================================================
# RAW gain 선택
# ============================================================
def _pick_per_candle_gain(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    return np.where(np.abs(up) >= np.abs(dn), up, dn).astype(np.float32)

# ============================================================
# RAW bin 생성 (🔥 동적 엣지 튜닝 + 꼬리 정리 버전)
# ============================================================
def _raw_bins(dist: np.ndarray, target_bins: int) -> np.ndarray:
    """
    동적 분포 기반 엣지 계산:
    - 클래스 개수(target_bins)는 최대한 유지
    - 한 클래스당 최소 샘플 수(_MIN_SAMPLES_PER_CLASS)를 고려해 bin 수 자동 조정
    - 분위수(quantile) 기반이라 극단 구간이 3%~28% 같은 미친 폭으로 커지지 않음
    - 추가: 상/하위 극단 꼬리를 소량(trim)해서, 비정상 값이 전체 구간을 찢어먹지 않게 함
    """
    # 유효 값만 사용
    if dist is None:
        return np.linspace(-0.01, 0.01, target_bins + 1).astype(float)

    dist = np.asarray(dist, dtype=float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return np.linspace(-0.01, 0.01, target_bins + 1).astype(float)

    n = dist.size

    # 🔥 극단 꼬리 소량 제거 (구간 계산용)
    # - LABEL_TAIL_TRIM_FRAC 비율(예: 0.005 → 상/하위 0.5%)만 잠깐 제외
    # - 남는 샘플이 너무 적으면(절반 이하 등) 그냥 원본 그대로 사용
    if _TAIL_TRIM_FRAC > 0.0 and n >= 100:
        try:
            q_low, q_high = np.quantile(
                dist,
                [_TAIL_TRIM_FRAC, 1.0 - _TAIL_TRIM_FRAC]
            )
            mask = (dist >= q_low) & (dist <= q_high)
            trimmed = dist[mask]
            # 최소 유지 조건: 전체의 70% 이상, 그리고 최소 샘플 수 확보
            if trimmed.size >= max(_MIN_LABEL_CLASSES * 2, int(0.7 * n)):
                dist = trimmed
                n = dist.size
        except Exception:
            # quantile 계산 실패 시 trimming 생략
            pass

    lo = float(np.min(dist))
    hi = float(np.max(dist))
    if not np.isfinite(lo):
        lo = -0.01
    if not np.isfinite(hi):
        hi = 0.01
    if hi <= lo:
        hi = lo + 1e-6

    # 데이터 양이 적으면 어차피 세분화가 안 되므로 균등 분할로 빠르게 리턴
    if n < _MIN_SAMPLES_PER_CLASS * 2:
        return np.linspace(
            lo,
            hi,
            max(_MIN_LABEL_CLASSES, min(target_bins, 4)) + 1,
        ).astype(float)

    # 데이터가 허용하는 최대 bin 수 (각 bin에 최소 샘플 수를 갖도록)
    max_bins_by_samples = max(1, n // max(1, _MIN_SAMPLES_PER_CLASS))

    # 실제 사용할 bin 수
    # - 너무 많지도 않고
    # - 최소 클래스 수는 지키면서
    bins = int(min(target_bins, max_bins_by_samples))
    bins = int(max(_MIN_LABEL_CLASSES, bins))

    if bins <= 1:
        # 어쩔 수 없이 1개 bin 수준이면 최소 2개로 쪼개 준다
        return np.linspace(lo, hi, 2 + 1).astype(float)

    # 분위수(quantile) 기반 엣지 계산
    # 예: bins=14 → q = [0, 1/14, 2/14, ..., 1]
    qs = np.linspace(0.0, 1.0, bins + 1)
    try:
        edges = np.quantile(dist, qs)
    except Exception:
        # quantile 실패시 기존 균등 분할로 fallback
        return np.linspace(lo, hi, bins + 1).astype(float)

    edges = np.asarray(edges, dtype=float)

    # 단조 증가(겹치지 않도록) 보장
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    # 안전장치: 전체 범위는 원래 lo~hi 를 벗어나지 않게 클램프
    edges[0] = min(edges[0], lo)
    edges[-1] = max(edges[-1], hi)

    return edges

def _vector_bin(gains: np.ndarray, edges: np.ndarray) -> np.ndarray:
    e = edges.copy()
    e[-1] += 1e-12
    bins = np.searchsorted(e, gains, side="right") - 1
    return np.clip(bins, 0, edges.size - 2).astype(np.int64)

# ============================================================
# 희소 클래스 병합 (아이디어 A) — 옵션
# ============================================================
def _merge_sparse_bins(edges: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    아이디어 A:
    - 한 "클래스"에 들어온 샘플 수가 너무 적으면,
      바로 이웃 bin과 합쳐서 극단 꼬리를 줄인다.
    - 기본은 MERGE_SPARSE_LABEL_BINS=0 으로 꺼져 있고,
      1로 켠 경우에만 실제 병합에 사용한다.
    반환: (새 edges, 새 bin_counts)
    """
    try:
        edges = np.asarray(edges, dtype=float)
        if edges.size < 3:
            return edges, np.zeros(max(0, edges.size - 1), dtype=int)

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return edges, np.zeros(max(0, edges.size - 1), dtype=int)

        # 초기 bin 카운트 계산 (라벨 기준 분포)
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

        # 이미 충분히 bin 수가 적으면 더 줄이지 않는다.
        if counts.size <= min_classes or max_passes == 0:
            return edges, counts

        e = edges.copy()
        c = counts.copy()

        for _ in range(max_passes):
            if c.size <= min_classes:
                break

            # 희소 bin 인덱스: 샘플 수가 기준 미만인 bin (클래스)
            sparse_idx = np.where(c < min_samples)[0]
            if sparse_idx.size == 0:
                break

            # 가장 빈약한 bin부터 처리
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
                if not left_ok and not right_ok:
                    continue

                # 양쪽 다 있으면 더 "두꺼운" 쪽으로 합치기
                if left_ok and right_ok:
                    if c[idx - 1] >= c[idx + 1]:
                        nbr = idx - 1
                    else:
                        nbr = idx + 1
                elif left_ok:
                    nbr = idx - 1
                else:
                    nbr = idx + 1

                # 실제 병합 수행 (인접 bin만)
                if nbr == idx - 1:
                    # 왼쪽과 병합: 경계 e[idx] 제거, c[nbr] += c[idx]
                    c[nbr] = c[nbr] + c[idx]
                    c = np.delete(c, idx)
                    e = np.delete(e, idx)
                elif nbr == idx + 1:
                    # 오른쪽과 병합: 경계 e[idx+1] 제거, c[idx] += c[nbr]
                    c[idx] = c[idx] + c[nbr]
                    c = np.delete(c, nbr)
                    e = np.delete(e, nbr)
                else:
                    # 논리상 올 일 없음
                    continue

                changed = True

            if not changed:
                break

        # 안전검사
        if e.size != c.size + 1:
            # 뭔가 꼬였으면 원본 유지 (라벨-경계 불일치 방지)
            logger.warning("merge_sparse_bins: edge/count size mismatch → 원본 유지")
            return edges, counts

        return e, c
    except Exception as ex:
        logger.warning("merge_sparse_bins failed: %s", ex)
        # 실패 시 원본 그대로 반환
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
# 수익률 계산 (핵심 수정)
# ============================================================
def compute_label_returns(df: pd.DataFrame, symbol: str, strategy: str):
    pure = _normalize_strategy_name(strategy)
    H = _get_fixed_horizon_candles(pure)   # 🔥 전략별 H 고정 적용 (단기/중기/장기 모두 1캔들)
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

    # 분포: up/dn 합친 전체 분포 기준으로 1차 엣지 생성
    dist = np.concatenate([dn_c, up_c], axis=0)

    # 1차: 동적 RAW bin 생성 (꼬리 정리 + quantile 기반)
    edges = _raw_bins(dist, target_bins)

    # 2차: 희소 bin 병합 (아이디어 A) — 옵션
    if MERGE_SPARSE_LABEL_BINS:
        edges, _ = _merge_sparse_bins(edges, gains)

    # 병합(또는 원본) 경계를 기준으로 최종 라벨링
    labels = _vector_bin(gains, edges)

    # 최종 bin 분포/폭 재계산 (gains 기준)
    edges2 = edges.copy()
    edges2[-1] += 1e-12
    bin_counts, _ = np.histogram(gains, bins=edges2)
    spans = np.diff(edges) * 100.0

    # 🔥 학습에서만 버릴 중앙(0% 근처) 마스크
    #  - |gain| < TRAIN_ZERO_BAND_ABS → train_mask = 0  (학습에서 제외)
    #  - 나머지 → train_mask = 1      (학습에 사용)
    train_mask = (np.abs(gains) >= float(TRAIN_ZERO_BAND_ABS)).astype(np.int8)

    sl = 0.02
    extra_cols = {
        "future_up": up_c,
        "future_dn": dn_c,
        "up_ge_2pct": (up_c >= sl).astype(np.int8),
        "dn_le_-2pct": (dn_c <= -sl).astype(np.int8),
        "train_mask": train_mask,  # 🔥 학습용 마스크 추가
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

    # up/dn 전체 분포
    dist = np.concatenate([dn, up], axis=0)
    target_bins = _auto_target_bins(len(df))

    # 1차: RAW bin (꼬리 정리 + quantile 기반)
    edges = _raw_bins(dist, target_bins)

    # gains 먼저 계산
    gains = _pick_per_candle_gain(up, dn)

    # 2차: 희소 bin 병합 (gains 기준) — 옵션
    if MERGE_SPARSE_LABEL_BINS:
        edges, _ = _merge_sparse_bins(edges, gains)

    labels = _vector_bin(gains, edges)

    edges2 = edges.copy()
    edges2[-1] += 1e-12
    bin_counts, _ = np.histogram(gains, bins=edges2)
    spans = np.diff(edges) * 100.0

    strategy = "단기" if horizon_hours <= 4 else ("중기" if horizon_hours <= 24 else "장기")

    # 🔥 학습에서만 버릴 중앙(0% 근처) 마스크
    train_mask = (np.abs(gains) >= float(TRAIN_ZERO_BAND_ABS)).astype(np.int8)

    extra_cols = {
        "future_up": up,
        "future_dn": dn,
        "up_ge_2pct": (up >= 0.02).astype(np.int8),
        "dn_le_-2pct": (dn <= -0.02).astype(np.int8),
        "train_mask": train_mask,  # 🔥 학습용 마스크 추가
    }

    _save_label_table(
        df, symbol, strategy, gains, labels,
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
        strategy,
        edges.astype(float),
        bin_counts.astype(int),
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
