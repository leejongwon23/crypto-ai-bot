# config.py (REVISED, 2025-09-12) — side-quantile bins, zero-band, min-per-bin, dedup guards

import json, os

CONFIG_PATH = "/persistent/config.json"

# ===============================
# 기본 설정 + 신규 옵션(기본 OFF)
# ===============================
_default_config = {
    "NUM_CLASSES": 20,
    "FEATURE_INPUT_SIZE": 24,
    "FAIL_AUGMENT_RATIO": 3,
    "MIN_FEATURES": 5,
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"],
    "SYMBOL_GROUP_SIZE": 3,

    # ✅ SSL 캐시 디렉토리
    "SSL_CACHE_DIR": "/persistent/ssl_models",

    # --- [2] 레짐(시장상태) 태깅 옵션 ---
    "REGIME": {
        "enabled": False,
        "lookback": 200,
        "atr_window": 14,
        "rsi_window": 14,
        "trend_window": 50,
        "vol_high_pct": 0.9,
        "vol_low_pct": 0.5,
        "cooldown_min": 5
    },

    # --- [3] 확률 캘리브레이션(스케일링) 옵션 ---
    "CALIB": {
        "enabled": False,
        "method": "platt",  # "platt" | "temperature"
        "min_samples": 500,
        "refresh_hours": 12,
        "per_model": True,
        "save_dir": "/persistent/calibration",
        "fallback_identity": True
    },

    # --- [5] 실패학습(하드 예시) 옵션 ---
    "FAILLEARN": {
        "enabled": False,
        "cooldown_min": 60,
        "max_samples": 1000,
        "class_weight_boost": 1.5,
        "min_return_abs": 0.003
    },

    # --- [NEW] 클래스 경계 생성기 세부 옵션 ---
    "CLASS_GEN": {
        "method": "quantile_side",   # "quantile_side" | "quantile" | "linear"
        "max_bins": 20,              # 상한(기본 20). 분포/데이터에 따라 10, 5 등으로 '자연' 축소 가능
        "min_bins": 4,               # 하한
        "group_size": 5,             # train.py의 그룹 슬라이싱과 동일
        "zero_band_eps": 0.0015,     # 중립 밴드 반폭(±0.15%p)
        "min_range_width": 0.0005,   # 최소 구간 폭(0.05%p) — 너무 촘촘하면 분류기 불안정
        "min_per_bin": 30,           # 각 bin 최소 샘플 수(중복 quantile 발생 시 병합 가이드)
        "dedup_eps": 1e-6            # 동일 엣지 중복시 살짝 벌리기
    },
}

# ✅ 전략별 K라인 설정
STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 1000, "binance_interval": "4h"},
    "중기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
    "장기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
}

# ✅ 전략별 수익률 캡(과장 방지용)
_STRATEGY_RETURN_CAP_POS_MAX = {"단기": 0.12, "중기": 0.25, "장기": 0.50}
_STRATEGY_RETURN_CAP_NEG_MIN = {"단기": -0.12, "중기": -0.25, "장기": -0.50}

# ✅ 표시 안정화용 파라미터
_MIN_RANGE_WIDTH   = 0.0005   # 0.05%
_ROUND_DECIMALS    = 4
_EPS_ZERO_BAND     = 0.0015   # ±0.15%p
_DISPLAY_MIN_RET   = 1e-4

_config = _default_config.copy()
_dynamic_num_classes = None
_ranges_cache = {}

def _quiet(): return os.getenv("QUIET_CONFIG_LOG", "0") == "1"
def _log(msg):
    if not _quiet():
        try: print(msg)
        except Exception: pass

def _deep_merge(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            if k not in dst:
                dst[k] = v

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
        _config = _loaded if isinstance(_loaded, dict) else _default_config.copy()
        _deep_merge(_config, _default_config)
        _log("[✅ config.py] config.json 로드/보강 완료")
    except Exception as e:
        _log(f"[⚠️ config.py] config.json 로드 실패 → 기본값 사용: {e}")
else:
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_default_config, f, ensure_ascii=False, indent=2)
        _log("[ℹ️ config.py] 기본 config.json 생성")
    except Exception as e:
        _log(f"[⚠️ config.py] 기본 config.json 생성 실패: {e}")

def save_config():
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_config, f, ensure_ascii=False, indent=2)
        _log("[✅ config.py] config.json 저장 완료")
    except Exception as e:
        _log(f"[⚠️ config.py] config.json 저장 실패 → {e}")

# ------------------------
# ✅ Binance 폴백 상태 로그
# ------------------------
try:
    _ENABLE_BINANCE = int(os.getenv("ENABLE_BINANCE", "1"))
    _log("[config] ENABLE_BINANCE=1 (fallback ready)" if _ENABLE_BINANCE == 1
         else "[config] ENABLE_BINANCE=0 (fallback disabled)")
except Exception:
    pass

# ------------------------
# Getter / Setter (기존)
# ------------------------
def set_NUM_CLASSES(n):
    global _dynamic_num_classes
    _dynamic_num_classes = int(n)

def get_NUM_CLASSES():
    global _dynamic_num_classes
    return _dynamic_num_classes if _dynamic_num_classes is not None else _config.get("NUM_CLASSES", _default_config["NUM_CLASSES"])

def get_FEATURE_INPUT_SIZE():
    return _config.get("FEATURE_INPUT_SIZE", _default_config["FEATURE_INPUT_SIZE"])

def get_FAIL_AUGMENT_RATIO():
    return _config.get("FAIL_AUGMENT_RATIO", _default_config["FAIL_AUGMENT_RATIO"])

def get_MIN_FEATURES():
    return _config.get("MIN_FEATURES", _default_config["MIN_FEATURES"])

def get_SYMBOLS():
    return _config.get("SYMBOLS", _default_config["SYMBOLS"])

def get_SYMBOL_GROUPS():
    symbols = get_SYMBOLS()
    group_size = _config.get("SYMBOL_GROUP_SIZE", _default_config["SYMBOL_GROUP_SIZE"])
    return [symbols[i:i+group_size] for i in range(0, len(symbols), group_size)]

def get_class_groups(num_classes=None, group_size=None):
    if num_classes is None or num_classes < 2:
        num_classes = get_NUM_CLASSES()
    if group_size is None:
        group_size = _config.get("CLASS_GEN", {}).get("group_size", 5)
    if num_classes <= group_size:
        groups = [list(range(num_classes))]
    else:
        groups = [list(range(i, min(i + group_size, num_classes))) for i in range(0, num_classes, group_size)]
    _log(f"[📊 클래스 그룹화] 총 클래스 수: {num_classes}, 그룹 크기: {group_size}, 그룹 개수: {len(groups)}")
    return groups

# ------------------------
# 신규 옵션 Getter (2·3·5)
# ------------------------
def get_REGIME():
    return _config.get("REGIME", _default_config["REGIME"])

def get_CALIB():
    return _config.get("CALIB", _default_config["CALIB"])

def get_FAILLEARN():
    return _config.get("FAILLEARN", _default_config["FAILLEARN"])

def get_CLASS_GEN():
    return _config.get("CLASS_GEN", _default_config["CLASS_GEN"])

# ------------------------
# 수익률 클래스 경계 유틸
# ------------------------
def _round2(x: float) -> float:
    return round(float(x), _ROUND_DECIMALS)

def _cap_by_strategy(x: float, strategy: str) -> float:
    pos_cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy)
    neg_cap = _STRATEGY_RETURN_CAP_NEG_MIN.get(strategy)
    if x > 0 and pos_cap is not None: return min(x, pos_cap)
    if x < 0 and neg_cap is not None: return max(x, neg_cap)
    return x

def _enforce_min_width(low: float, high: float, min_w: float):
    if (high - low) < min_w:
        high = low + min_w
    return low, high

def _strategy_horizon_hours(strategy: str) -> int:
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)

def _future_extreme_signed_returns(df, horizon_hours: int):
    import numpy as np, pandas as pd
    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    close = df["close"].astype(float).values
    high = (df["high"] if "high" in df.columns else df["close"]).astype(float).values
    low  = (df["low"]  if "low"  in df.columns else df["close"]).astype(float).values
    ts = (ts.dt.tz_localize("UTC") if getattr(ts.dt,"tz",None) is None else ts).dt.tz_convert("Asia/Seoul")
    horizon = pd.Timedelta(hours=int(horizon_hours))
    up = np.zeros(len(df), dtype=np.float32); dn = np.zeros(len(df), dtype=np.float32)
    j_up = j_dn = 0
    for i in range(len(df)):
        t0 = ts.iloc[i]; t1 = t0 + horizon
        j = max(j_up, i); max_h = high[i]
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > max_h: max_h = high[j]
            j += 1
        j_up = max(j_up, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        up[i] = float((max_h - base) / (base + 1e-12))
        k = max(j_dn, i); min_l = low[i]
        while k < len(df) and ts.iloc[k] <= t1:
            if low[k] < min_l: min_l = low[k]
            k += 1
        j_dn = max(j_dn, i)
        dn[i] = float((min_l - base) / (base + 1e-12))
    signed = np.concatenate([dn, up]).astype(np.float32)
    return signed

def get_class_return_range(class_id: int, symbol: str, strategy: str):
    key = (symbol, strategy)
    ranges = _ranges_cache.get(key)
    if ranges is None:
        ranges = get_class_ranges(symbol=symbol, strategy=strategy)
        _ranges_cache[key] = ranges
    n = len(ranges)
    if not (0 <= class_id < n):
        raise ValueError(f"class_id {class_id} 범위 오류 (0~{n-1})")
    return ranges[class_id]

def class_to_expected_return(class_id: int, symbol: str, strategy: str):
    r_min, r_max = get_class_return_range(class_id, symbol, strategy)
    return (r_min + r_max) / 2

def get_class_ranges(symbol=None, strategy=None, method=None, group_id=None, group_size=None):
    """
    분포 기반 가변 클래스 생성:
      - 기본: "quantile_side" (음/양 분리 양측 분위수) + 중앙 중립밴드 보장
      - 데이터 부족/중복 엣지 → 안전한 균등분할로 폴백
      - 클래스 수는 ‘자연 축소/유지’만 허용(강제 축소 없음)
    """
    import numpy as np
    from data.utils import get_kline_by_strategy

    GEN = get_CLASS_GEN()
    MAX_CLASSES = int(GEN.get("max_bins", 20))
    MIN_CLASSES = int(GEN.get("min_bins", 4))
    ZERO_EPS    = float(GEN.get("zero_band_eps", _EPS_ZERO_BAND))
    MIN_W       = float(GEN.get("min_range_width", _MIN_RANGE_WIDTH))
    MIN_PER_BIN = int(GEN.get("min_per_bin", 30))
    DEDUP_EPS   = float(GEN.get("dedup_eps", 1e-6))

    if method is None:
        method = GEN.get("method", "quantile_side")

    def _fix_monotonic(ranges):
        fixed=[]; prev_hi=None
        for lo,hi in ranges:
            if prev_hi is not None and lo < prev_hi:
                lo = prev_hi
                lo,hi = _enforce_min_width(lo,hi,MIN_W)
            lo,hi = _round2(lo), _round2(hi)
            if hi <= lo: hi = _round2(lo + MIN_W)
            fixed.append((lo,hi)); prev_hi=hi
        return fixed

    def _ensure_zero_band(ranges):
        crosses=[i for i,(lo,hi) in enumerate(ranges) if lo < 0.0 <= hi]
        if crosses:
            i=crosses[0]; lo,hi=ranges[i]
            if (hi-lo) < max(MIN_W, ZERO_EPS*2):
                lo=min(lo,-ZERO_EPS); hi=max(hi,ZERO_EPS)
                ranges[i]=(_round2(lo), _round2(hi))
            return ranges
        left_idx  = max([i for i,(lo,hi) in enumerate(ranges) if hi <= 0.0], default=None)
        right_idx = min([i for i,(lo,hi) in enumerate(ranges) if lo >  0.0], default=None)
        if left_idx is None or right_idx is None: return ranges
        lo_l,hi_l=ranges[left_idx]; lo_r,hi_r=ranges[right_idx]
        ranges[left_idx]  = (_round2(lo_l), _round2(-ZERO_EPS))
        ranges[right_idx] = (_round2( ZERO_EPS), _round2(hi_r))
        ranges = ranges[:right_idx] + [(_round2(-ZERO_EPS), _round2(ZERO_EPS))] + ranges[right_idx:]
        return _fix_monotonic(ranges)

    def compute_equal_ranges(n_cls, reason=""):
        n_cls=max(MIN_CLASSES, int(n_cls)); n_cls=min(MAX_CLASSES, n_cls)
        neg=_STRATEGY_RETURN_CAP_NEG_MIN.get(strategy, -0.5)
        pos=_STRATEGY_RETURN_CAP_POS_MAX.get(strategy,  0.5)
        step=(pos-neg)/n_cls
        raw=[(neg+i*step, neg+(i+1)*step) for i in range(n_cls)]
        ranges=[]
        for lo,hi in raw:
            lo,hi=_enforce_min_width(lo,hi,MIN_W)
            lo,hi=_cap_by_strategy(lo, strategy), _cap_by_strategy(hi, strategy)
            ranges.append((_round2(lo), _round2(hi)))
        if reason: _log(f"[⚠️ 균등 분할 클래스 사용] 사유: {reason}")
        ranges=_fix_monotonic(ranges)
        return _ensure_zero_band(ranges)

    def _dedup_edges(edges: np.ndarray):
        """동일/역전 엣지 보정(아주 작게 벌림)"""
        e = edges.copy()
        for i in range(1, len(e)):
            if e[i] <= e[i-1]:
                e[i] = e[i-1] + DEDUP_EPS
        return e

    def _allocate_side_bins(n_total, n_neg, n_pos):
        # 중립 1칸 보장 목표로 양/음에 비례 분배, 최소 1칸
        n_middle = 1
        remain = max(MIN_CLASSES, n_total)
        remain = min(MAX_CLASSES, remain)
        remain_side = max(0, remain - n_middle)
        tot = max(1, n_neg + n_pos)
        left = max(1, int(round(remain_side * (n_neg / tot))))
        right = max(1, remain_side - left)
        return left, n_middle, right

    def compute_ranges_from_kline():
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 30 or "close" not in df:
                return compute_equal_ranges(10, reason="가격 데이터 부족")

            horizon_hours = _strategy_horizon_hours(strategy)
            rets = _future_extreme_signed_returns(df, horizon_hours=horizon_hours)
            rets = rets[np.isfinite(rets)]
            if rets.size < 10:
                return compute_equal_ranges(10, reason="수익률 샘플 부족")

            # 전략별 ±캡
            rets = np.array([_cap_by_strategy(float(r), strategy) for r in rets], dtype=np.float32)

            base_n = int(_config.get("NUM_CLASSES", 20))
            target_n = max(MIN_CLASSES, min(MAX_CLASSES, base_n))

            if method == "linear":
                return compute_equal_ranges(target_n, reason="linear 요청")

            if method in ("quantile_side", "quantile"):
                # 중복 엣지/희소 구간 방지용: 최소 per-bin 샘플 확인
                neg = rets[rets < 0.0]; pos = rets[rets > 0.0]
                n_neg, n_pos = int(neg.size), int(pos.size)

                if method == "quantile_side" and n_neg + n_pos >= MIN_CLASSES * MIN_PER_BIN:
                    # 양/음 분리 분위수
                    left_bins, mid_bins, right_bins = _allocate_side_bins(target_n, n_neg, n_pos)
                    # 최소 샘플/빈 고려하여 bins 재조정
                    while left_bins>1 and n_neg/left_bins < MIN_PER_BIN: left_bins -= 1
                    while right_bins>1 and n_pos/right_bins < MIN_PER_BIN: right_bins -= 1
                    # 총합 맞추기(중립 1)
                    total_bins = left_bins + mid_bins + right_bins
                    if total_bins < MIN_CLASSES:
                        add = MIN_CLASSES - total_bins
                        # 샘플 더 많은 쪽에 우선 할당
                        if n_neg >= n_pos: left_bins += add
                        else: right_bins += add
                    # 엣지 계산
                    neg_edges = np.quantile(neg, np.linspace(0, 1, left_bins + 1)) if left_bins>0 else np.array([])
                    pos_edges = np.quantile(pos, np.linspace(0, 1, right_bins + 1)) if right_bins>0 else np.array([])
                    neg_edges = _dedup_edges(neg_edges) if neg_edges.size else neg_edges
                    pos_edges = _dedup_edges(pos_edges) if pos_edges.size else pos_edges

                    cooked=[]
                    # 음수 구간(상승 순서대로)
                    for i in range(max(0, left_bins)):
                        lo, hi = float(neg_edges[i]), float(neg_edges[i+1])
                        lo, hi = _cap_by_strategy(lo, strategy), _cap_by_strategy(hi, strategy)
                        lo, hi = _enforce_min_width(lo, hi, MIN_W)
                        cooked.append((_round2(lo), _round2(hi)))
                    # 중립 한칸
                    cooked.append((_round2(-ZERO_EPS), _round2(ZERO_EPS)))
                    # 양수 구간
                    for i in range(max(0, right_bins)):
                        lo, hi = float(pos_edges[i]), float(pos_edges[i+1])
                        lo, hi = _cap_by_strategy(lo, strategy), _cap_by_strategy(hi, strategy)
                        lo, hi = _enforce_min_width(lo, hi, MIN_W)
                        cooked.append((_round2(lo), _round2(hi)))

                else:
                    # 단일 분포 분위수(표준)
                    edges = np.quantile(rets, np.linspace(0, 1, target_n + 1))
                    edges = _dedup_edges(edges)
                    cooked=[]
                    for i in range(target_n):
                        lo, hi = float(edges[i]), float(edges[i+1])
                        lo, hi = _cap_by_strategy(lo, strategy), _cap_by_strategy(hi, strategy)
                        lo, hi = _enforce_min_width(lo, hi, MIN_W)
                        cooked.append((_round2(lo), _round2(hi)))

                fixed = _fix_monotonic(cooked)
                fixed = _ensure_zero_band(fixed)

                # 안전 가드
                if not fixed or len(fixed) < 2:
                    return compute_equal_ranges(max(10, MIN_CLASSES), reason="최종 경계 부족(가드)")

                return fixed

            # 미지정 → 균등
            return compute_equal_ranges(target_n, reason="unknown method")

        except Exception as e:
            return compute_equal_ranges(10, reason=f"예외 발생: {e}")

    all_ranges = compute_ranges_from_kline()

    if symbol is not None and strategy is not None:
        _ranges_cache[(symbol, strategy)] = all_ranges

    # 디버그(요약)
    try:
        if symbol is not None and strategy is not None and not _quiet():
            import numpy as np
            from data.utils import get_kline_by_strategy as _get_kline_dbg
            df_price_dbg = _get_kline_dbg(symbol, strategy)
            if df_price_dbg is not None and len(df_price_dbg) >= 2 and "close" in df_price_dbg:
                horizon_hours = _strategy_horizon_hours(strategy)
                rets_dbg = _future_extreme_signed_returns(df_price_dbg, horizon_hours=horizon_hours)
                rets_dbg = rets_dbg[np.isfinite(rets_dbg)]
                if rets_dbg.size > 0:
                    rets_dbg = np.array([_cap_by_strategy(float(r), strategy) for r in rets_dbg], dtype=np.float32)
                    qs = np.quantile(rets_dbg, [0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00])
                    def _r2(z): return round(float(z), _ROUND_DECIMALS)
                    print(f"[📈 수익률분포(±)] {symbol}-{strategy} "
                          f"min={_r2(qs[0])}, p25={_r2(qs[1])}, p50={_r2(qs[2])}, "
                          f"p75={_r2(qs[3])}, p90={_r2(qs[4])}, p95={_r2(qs[5])}, "
                          f"p99={_r2(qs[6])}, max={_r2(qs[7])}")
                    print(f"[📏 클래스경계 로그] {symbol}-{strategy} → {len(all_ranges)}개")
                    print(f"[📏 경계 리스트] {symbol}-{strategy} → {all_ranges}")
                    edges = [all_ranges[0][0]] + [hi for (_, hi) in all_ranges]
                    edges[-1] = float(edges[-1]) + 1e-9
                    hist, _ = np.histogram(rets_dbg, bins=edges)
                    print(f"[📐 클래스 분포] {symbol}-{strategy} count={int(hist.sum())} → {hist.tolist()}")
            else:
                print(f"[ℹ️ 수익률분포 스킵] {symbol}-{strategy} → 데이터 부족")
    except Exception as _e:
        _log(f"[⚠️ 디버그 로그 실패] {symbol}-{strategy} → {_e}")

    # ✅ 동적 클래스 수 반영 (훈련 파이프라인과 합치)
    try:
        if isinstance(all_ranges, list) and len(all_ranges) >= 2:
            set_NUM_CLASSES(len(all_ranges))
    except Exception:
        pass

    if group_id is None:
        return all_ranges

    # ▶ 그룹 슬라이싱
    if group_size is None:
        group_size = int(get_CLASS_GEN().get("group_size", 5))
    start = int(group_id) * int(group_size)
    end = start + int(group_size)
    if start >= len(all_ranges):
        return []
    return all_ranges[start:end]

# ------------------------
# 🔧 환경변수 기반 퍼포먼스/학습 토글
# ------------------------
def _get_int(name, default):
    try: return int(os.getenv(name, str(default)))
    except Exception: return int(default)

def _get_float(name, default):
    try: return float(os.getenv(name, str(default)))
    except Exception: return float(default)

CPU_THREADS        = _get_int("OMP_NUM_THREADS", 4)
TRAIN_NUM_WORKERS  = _get_int("TRAIN_NUM_WORKERS", 2)
TRAIN_BATCH_SIZE   = _get_int("TRAIN_BATCH_SIZE", 256)
ORDERED_TRAIN      = _get_int("ORDERED_TRAIN", 1)

# ⚠️ ‘표시’ 하한만 책임. 실제 타겟 계산은 모델/서비스 로직에서 결정.
PREDICT_MIN_RETURN = _get_float("PREDICT_MIN_RETURN", 0.0)
DISPLAY_MIN_RETURN = _get_float("DISPLAY_MIN_RETURN", _DISPLAY_MIN_RET)

SSL_CACHE_DIR      = os.getenv("SSL_CACHE_DIR", _default_config["SSL_CACHE_DIR"])

def get_CPU_THREADS():        return CPU_THREADS
def get_TRAIN_NUM_WORKERS():  return TRAIN_NUM_WORKERS
def get_TRAIN_BATCH_SIZE():   return TRAIN_BATCH_SIZE
def get_ORDERED_TRAIN():      return ORDERED_TRAIN
def get_PREDICT_MIN_RETURN(): return PREDICT_MIN_RETURN
def get_DISPLAY_MIN_RETURN(): return DISPLAY_MIN_RETURN
def get_SSL_CACHE_DIR():      return os.getenv("SSL_CACHE_DIR", _config.get("SSL_CACHE_DIR", _default_config["SSL_CACHE_DIR"]))

# ------------------------
# 전역 캐시된 값(기존)
# ------------------------
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
NUM_CLASSES        = get_NUM_CLASSES()
FAIL_AUGMENT_RATIO = get_FAIL_AUGMENT_RATIO()
MIN_FEATURES       = get_MIN_FEATURES()

__all__ = [
    "STRATEGY_CONFIG",
    "get_NUM_CLASSES", "set_NUM_CLASSES",
    "get_FEATURE_INPUT_SIZE",
    "get_class_groups", "get_class_ranges",
    "get_class_return_range", "class_to_expected_return",
    "get_SYMBOLS", "get_SYMBOL_GROUPS",
    "get_REGIME", "get_CALIB", "get_FAILLEARN", "get_CLASS_GEN",
    "get_CPU_THREADS", "get_TRAIN_NUM_WORKERS", "get_TRAIN_BATCH_SIZE",
    "get_ORDERED_TRAIN", "get_PREDICT_MIN_RETURN", "get_DISPLAY_MIN_RETURN",
    "get_SSL_CACHE_DIR",
    "FEATURE_INPUT_SIZE", "NUM_CLASSES", "FAIL_AUGMENT_RATIO", "MIN_FEATURES",
        ]
