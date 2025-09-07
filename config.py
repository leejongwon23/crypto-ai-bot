# config.py (FIXED, 2025-09-08)

import json
import os

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
}

# ✅ 전략별 K라인 설정
STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 1000, "binance_interval": "4h"},
    "중기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
    "장기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
}

# ✅ 전략별 양의 수익률 상한(과장 방지용 캡)
_STRATEGY_RETURN_CAP_POS_MAX = {
    "단기": 0.12,
    "중기": 0.25,
    "장기": 0.50,
}

# ✅ 표시 안정화용 파라미터
_MIN_RANGE_WIDTH   = 0.0005   # 0.05%
_ROUND_DECIMALS    = 4        # 소수 넷째 자리
_EPS_START         = 1e-4     # 첫 구간이 0%로 딱 붙는 걸 회피
_DISPLAY_MIN_RET   = 1e-4     # 표시에만 쓰는 최소값(0.01%p)

_config = _default_config.copy()
_dynamic_num_classes = None
_ranges_cache = {}

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
        print("[✅ config.py] config.json 로드/보강 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 로드 실패 → 기본값 사용: {e}")
else:
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_default_config, f, ensure_ascii=False, indent=2)
        print("[ℹ️ config.py] 기본 config.json 생성")
    except Exception as e:
        print(f"[⚠️ config.py] 기본 config.json 생성 실패: {e}")

def save_config():
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_config, f, ensure_ascii=False, indent=2)
        print("[✅ config.py] config.json 저장 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 저장 실패 → {e}")

# ------------------------
# ✅ Binance 폴백 상태 로그
# ------------------------
try:
    _ENABLE_BINANCE = int(os.getenv("ENABLE_BINANCE", "1"))
    if _ENABLE_BINANCE == 1:
        print("[config] ENABLE_BINANCE=1 (fallback ready)")
    else:
        print("[config] ENABLE_BINANCE=0 (fallback disabled)")
except Exception:
    pass

# ------------------------
# Getter / Setter (기존)
# ------------------------
def set_NUM_CLASSES(n):
    global _dynamic_num_classes
    _dynamic_num_classes = n

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

def get_class_groups(num_classes=None, group_size=5):
    if num_classes is None or num_classes < 2:
        num_classes = get_NUM_CLASSES()
    if num_classes <= group_size:
        groups = [list(range(num_classes))]
    else:
        groups = [list(range(i, min(i + group_size, num_classes))) for i in range(0, num_classes, group_size)]
    print(f"[📊 클래스 그룹화] 총 클래스 수: {num_classes}, 그룹 크기: {group_size}, 그룹 개수: {len(groups)}")
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

# ------------------------
# 수익률 클래스 경계 유틸
# ------------------------
def _round2(x: float) -> float:
    return round(float(x), _ROUND_DECIMALS)

def _cap_positive_by_strategy(x: float, strategy: str) -> float:
    cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy, None)
    if cap is not None and x > 0:
        return min(x, cap)
    return x

def _enforce_min_width(low: float, high: float):
    if (high - low) < _MIN_RANGE_WIDTH:
        high = low + _MIN_RANGE_WIDTH
    return low, high

def _strategy_horizon_hours(strategy: str) -> int:
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)

def _future_max_high_return_series(df, horizon_hours: int):
    import numpy as np
    import pandas as pd

    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    close = df["close"].astype(float).values
    high = (df["high"] if "high" in df.columns else df["close"]).astype(float).values

    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")

    out = np.zeros(len(df), dtype=np.float32)
    horizon = pd.Timedelta(hours=int(horizon_hours))

    j_start = 0
    for i in range(len(df)):
        t0 = ts.iloc[i]; t1 = t0 + horizon
        j = max(j_start, i)
        max_h = high[i]
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > max_h:
                max_h = high[j]
            j += 1
        j_start = max(j_start, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        out[i] = float((max_h - base) / (base + 1e-12))
    return out.astype(np.float32)

def get_class_return_range(class_id: int, symbol: str, strategy: str):
    key = (symbol, strategy)
    ranges = _ranges_cache.get(key)
    if ranges is None:
        ranges = get_class_ranges(symbol=symbol, strategy=strategy)
        _ranges_cache[key] = ranges
    assert 0 <= class_id < len(ranges), f"class_id {class_id} 범위 오류 (0~{len(ranges)-1})"
    return ranges[class_id]

def class_to_expected_return(class_id: int, symbol: str, strategy: str):
    r_min, r_max = get_class_return_range(class_id, symbol, strategy)
    return (r_min + r_max) / 2

def get_class_ranges(symbol=None, strategy=None, method="quantile", group_id=None, group_size=5):
    """
    미래 최대고가 수익률 기반 클래스 경계
    - r_i = (max(high[i..i+h]) - close[i]) / close[i]
    - 전략별 양수 캡
    - 최소 구간 폭
    - 모든 경계 소수 넷째 자리 반올림
    - 첫 구간을 0에서 살짝 띄워 표시상 0% 고정 현상 방지
    """
    import numpy as np
    from data.utils import get_kline_by_strategy

    MAX_CLASSES = 20

    def compute_equal_ranges(n_cls, reason=""):
        n_cls = max(4, int(n_cls))
        cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy, 0.5)
        start = _EPS_START  # 0이 아닌 작은 양수
        step = (float(cap) - start) / n_cls
        raw = [(start + i * step, start + (i + 1) * step) for i in range(n_cls)]
        ranges = []
        for lo, hi in raw:
            lo, hi = _enforce_min_width(lo, hi)
            ranges.append((_round2(lo), _round2(hi)))
        if reason:
            print(f"[⚠️ 균등 분할 클래스 사용] 사유: {reason}")
        return _fix_monotonic(ranges)

    def _fix_monotonic(ranges):
        fixed = []
        prev_hi = None
        for lo, hi in ranges:
            if prev_hi is not None and lo < prev_hi:
                lo = prev_hi
                lo, hi = _enforce_min_width(lo, hi)
            lo, hi = _round2(lo), _round2(hi)
            if hi <= lo:
                hi = _round2(lo + _MIN_RANGE_WIDTH)
            fixed.append((lo, hi))
            prev_hi = hi
        # 첫 구간이 0 또는 0에 매우 근접하면 EPS로 끌어올림(표시 안정)
        if fixed and fixed[0][0] <= 0.0:
            fixed[0] = (_round2(_EPS_START), fixed[0][1])
        return fixed

    def compute_split_ranges_from_kline():
        try:
            df_price = get_kline_by_strategy(symbol, strategy)
            if df_price is None or len(df_price) < 30 or "close" not in df_price:
                return compute_equal_ranges(10, reason="가격 데이터 부족")

            horizon_hours = _strategy_horizon_hours(strategy)
            rets = _future_max_high_return_series(df_price, horizon_hours=horizon_hours)
            rets = rets[np.isfinite(rets)]
            if rets.size < 10:
                return compute_equal_ranges(10, reason="수익률 샘플 부족")

            rets = np.maximum(rets, 0.0)

            cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy)
            if cap is not None and rets.size > 0:
                rets = np.minimum(rets, cap)

            base_n = int(_config.get("NUM_CLASSES", 20))
            n_cls = min(MAX_CLASSES, max(4, base_n))

            if method == "quantile":
                qs = np.quantile(rets, np.linspace(0, 1, n_cls + 1))
                # 첫 엣지가 0이면 EPS로 승격(표시 0% 회피)
                qs[0] = max(float(qs[0]), _EPS_START)
            else:
                lo0 = max(float(rets.min()), _EPS_START)
                qs = np.linspace(lo0, float(rets.max()), n_cls + 1)

            cooked = []
            for i in range(n_cls):
                lo, hi = float(qs[i]), float(qs[i + 1])
                lo, hi = _enforce_min_width(lo, hi)
                lo = _cap_positive_by_strategy(lo, strategy)
                hi = _cap_positive_by_strategy(hi, strategy)
                lo, hi = _round2(lo), _round2(hi)
                if hi <= lo:
                    hi = _round2(lo + _MIN_RANGE_WIDTH)
                cooked.append((lo, hi))

            fixed = _fix_monotonic(cooked)

            if not fixed or len(fixed) < 2:
                return compute_equal_ranges(10, reason="최종 경계 부족(가드)")

            return fixed

        except Exception as e:
            return compute_equal_ranges(10, reason=f"예외 발생: {e}")

    all_ranges = compute_split_ranges_from_kline()

    if symbol is not None and strategy is not None:
        _ranges_cache[(symbol, strategy)] = all_ranges

    # 디버그 로깅
    try:
        if symbol is not None and strategy is not None:
            import numpy as np
            from data.utils import get_kline_by_strategy as _get_kline_dbg

            df_price_dbg = _get_kline_dbg(symbol, strategy)
            if df_price_dbg is not None and len(df_price_dbg) >= 2 and "close" in df_price_dbg:
                horizon_hours = _strategy_horizon_hours(strategy)
                rets_dbg = _future_max_high_return_series(df_price_dbg, horizon_hours=horizon_hours)
                rets_dbg = rets_dbg[np.isfinite(rets_dbg)]
                if rets_dbg.size > 0:
                    rets_dbg = np.maximum(rets_dbg, 0.0)
                    cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy)
                    if cap is not None:
                        rets_dbg = np.minimum(rets_dbg, cap)

                    qs = np.quantile(rets_dbg, [0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00])
                    def _r2(z): return round(float(z), _ROUND_DECIMALS)
                    print(
                        f"[📈 수익률분포] {symbol}-{strategy} "
                        f"min={_r2(qs[0])}, p25={_r2(qs[1])}, p50={_r2(qs[2])}, "
                        f"p75={_r2(qs[3])}, p90={_r2(qs[4])}, p95={_r2(qs[5])}, "
                        f"p99={_r2(qs[6])}, max={_r2(qs[7])}"
                    )
                    print(f"[📏 클래스경계 로그] {symbol}-{strategy} → {len(all_ranges)}개")
                    print(f"[📏 경계 리스트] {symbol}-{strategy} → {all_ranges}")

                    edges = [all_ranges[0][0]] + [hi for (_, hi) in all_ranges]
                    edges[-1] = float(edges[-1]) + 1e-9
                    hist, _ = np.histogram(rets_dbg, bins=edges)
                    print(f"[📐 클래스 분포] {symbol}-{strategy} count={int(hist.sum())} → {hist.tolist()}")
            else:
                print(f"[ℹ️ 수익률분포 스킵] {symbol}-{strategy} → 데이터 부족")
    except Exception as _e:
        print(f"[⚠️ 디버그 로그 실패] {symbol}-{strategy} → {_e}")

    # ✅ 동적 클래스 수 반영
    try:
        if isinstance(all_ranges, list) and len(all_ranges) >= 2:
            set_NUM_CLASSES(len(all_ranges))
    except Exception:
        pass

    if group_id is None:
        return all_ranges
    return all_ranges[group_id * group_size: (group_id + 1) * group_size]

# ------------------------
# 🔧 환경변수 기반 퍼포먼스/학습 토글
# ------------------------
def _get_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

def _get_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

CPU_THREADS        = _get_int("OMP_NUM_THREADS", 4)
TRAIN_NUM_WORKERS  = _get_int("TRAIN_NUM_WORKERS", 2)
TRAIN_BATCH_SIZE   = _get_int("TRAIN_BATCH_SIZE", 256)
ORDERED_TRAIN      = _get_int("ORDERED_TRAIN", 1)

# ⚠️ ‘표시’ 하한만 책임. 실제 타겟 계산은 모델/서비스 로직에서 결정.
PREDICT_MIN_RETURN = _get_float("PREDICT_MIN_RETURN", 0.0)  # 강제 하한 제거
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
NUM_CLASSES = get_NUM_CLASSES()
FAIL_AUGMENT_RATIO = get_FAIL_AUGMENT_RATIO()
MIN_FEATURES = get_MIN_FEATURES()
