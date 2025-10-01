# config.py (STEP 5 FINAL+) — publish filter/ENV 확장 + 데이터 병합/클래스 일관성/CV 가드 옵션 추가
# - 원본 보존 + 옵션/헬퍼 확장: DATA / CLASS_ENFORCE / CV_CONFIG / RUNTIME getters
import json
import os

CONFIG_PATH = "/persistent/config.json"

# ===============================
# 기본 설정 + 신규 옵션
# ===============================
_default_config = {
    "NUM_CLASSES": 10,
    "MAX_CLASSES": 12,
    "FEATURE_INPUT_SIZE": 24,
    "FAIL_AUGMENT_RATIO": 3,
    "MIN_FEATURES": 5,
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"],
    "SYMBOL_GROUP_SIZE": 3,

    # ✅ 패턴/블렌딩 파라미터(후속 단계에서 사용)
    "PATTERN": {"K": 200},
    "BLEND":   {"alpha": 0.6, "beta": 0.2, "gamma": 0.2, "enabled": True},

    # ✅ SSL 캐시 디렉토리
    "SSL_CACHE_DIR": "/persistent/ssl_models",

    # --- [DATA] 거래소 병합/정합 옵션 (신설) ---
    "DATA": {
        "merge_enabled": True,                       # Bybit+Binance 병합 기본 ON
        "sources": ["bybit", "binance"],            # 우선순위는 prefer가 결정
        "prefer": "binance_if_overlap",             # 겹치면 binance 우선
        "align": {"method": "timestamp", "tolerance_sec": 60},
        "fill":  {"method": "ffill", "max_gap": 2}, # 2개 캔들까지 전방 보간 허용
        "dedup": {"enabled": True, "keep": "last"}
    },

    # --- [CLASS_ENFORCE] 그룹/심볼 간 클래스 수 일관성 (신설) ---
    "CLASS_ENFORCE": {
        "same_across_groups": True,   # 그룹0/1/2가 항상 동일 클래스 수 사용
        "same_across_symbols": True,  # 다른 심볼도 동일 클래스 수 사용
        "n_override": None            # 지정 시 동적결정 무시하고 이 값 고정
    },

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
        "enabled": True,
        "method": "temperature",
        "min_samples": 200,
        "refresh_hours": 12,
        "per_model": True,
        "save_dir": "/persistent/calibration",
        "fallback_identity": True
    },

    # --- [LOSS] 손실/가중치 옵션 ---
    "LOSS": {
        "use_focal": False,
        "alpha_mode": "auto",
        "label_smoothing": 0.02,
        "focal_gamma": 0.0,
        "class_weight": {
            "mode": "inverse_freq_clip",
            "min": 0.5,
            "max": 2.0
        }
    },

    # --- [AUG] 입력 증강 토글 ---
    "AUG": {"mixup": 0.0, "cutmix": 0.0},

    # --- [EVAL] 평가 설정 ---
    "EVAL": {"macro_f1": True, "topk": [1, 3], "use_cost_sensitive_argmax": True},

    # --- [5] 실패학습(하드 예시) 옵션 ---
    "FAILLEARN": {
        "enabled": False,
        "cooldown_min": 60,
        "max_samples": 1000,
        "class_weight_boost": 1.5,
        "min_return_abs": 0.003
    },

    # --- [Q] 품질 컷(모의고사 합격선) ---
    "QUALITY": {"VAL_F1_MIN": 0.20, "VAL_ACC_MIN": 0.20},

    # --- [BIN] 클래스 경계/병합 파라미터 ---
    #   method 기본값 "quantile" (실제 분포 기반), merge_sparse 기본 False
    "CLASS_BIN": {
        "method": "quantile",     # "fixed_step" | "quantile" | "linear"
        "strict": True,
        "zero_band_eps": 0.0020,  # ±0.20%p
        "min_width": 0.0010,      # 최소 폭 0.10%p
        "step_pct": 0.0050,       # (fixed_step일 때) 0.5% 단위
        "merge_sparse": {
            "enabled": False,
            "min_ratio": 0.01,    # 샘플 비율 임계
            "min_count_floor": 20,
            "prefer": "denser"
        }
    },

    # --- [CV_CONFIG] 교차검증·가드 (신설) ---
    "CV_CONFIG": {
        "folds": 5,                   # 기본 폴드 수 (ENV로도 제어)
        "min_per_class": 3,           # 각 폴드당 최소 클래스 샘플
        "fallback_reduce_folds": True,# 불가 시 자동 폴드 축소
        "fallback_stratified": True   # Stratified 실패 시 일반 KFold 폴백
    },

    # --- [TRAIN] 학습 스케줄/조기종료 ---
    "TRAIN": {
        "early_stop": {"patience": 4, "min_delta": 0.0005, "warmup_epochs": 2},
        "lr_scheduler": {"patience": 3, "min_lr": 5e-6},
        "ensure_class_coverage": True  # 각 epoch 검증세트 클래스커버리지 보장 시도
    },

    # --- [ENSEMBLE] 멀티-윈도우 앙상블 ---
    "ENSEMBLE": {"topk_windows": 3, "use_var_weight": True},

    # --- [SCHED] 학습 스케줄러 힌트 ---
    "SCHED": {
        "round_robin": True,
        "max_minutes_per_symbol": 10,
        "on_incomplete": "skip_and_rotate",
        "eval_during_training": True
    },

    # --- [PUBLISH] 텔레그램(또는 알림) 발송 필터 임계치 ---
    "PUBLISH": {
        "enabled": True,
        "recent_window": 10,
        "recent_success_min": 0.60,
        "min_expected_return": 0.010,
        "abstain_prob_min": 0.35,
        "min_meta_confidence": 0.0,
        "allow_shadow": True,
        "always_log": True
    },

    # --- ✅ [EVAL_RUNTIME] 평가 실행 주기/그레이스/슬랙(UTC 기준 저장) ---
    "EVAL_RUNTIME": {
        "timebase": "utc",
        "check_interval_min": 2,
        "grace_min": 5,
        "price_window_slack_min": 10,
        "max_backfill_hours": 48
    },

    # --- [ONCHAIN] 온체인 지표 옵션 (신설) ---
    "ONCHAIN": {
        "enabled": False,                # 기본 OFF (데이터 준비 후 ON)
        "dir": "/persistent/onchain",    # 온체인 CSV/파케 디렉토리
        "features": [                    # 기본 제공 키워드(임의 확장 가능)
            "active_address",
            "tx_count",
            "exchange_inflow",
            "exchange_outflow"
        ],
        "fill": {"method": "ffill", "max_gap": 6},  # 6 step까지 보간 허용
        "zscore_window": 96,             # 표준화 윈도우(시장/상관과 일관)
    },
}

# ✅ 전략별 K라인 설정 (모두 1200개로 통일)
STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 1200, "binance_interval": "4h"},
    "중기": {"interval": "D",   "limit": 1200, "binance_interval": "1d"},
    "장기": {"interval": "D",   "limit": 1200, "binance_interval": "1d"},
}

# ✅ 전략별 수익률 캡(과장 방지용)
_STRATEGY_RETURN_CAP_POS_MAX = {"단기": 0.12, "중기": 0.25, "장기": 0.50}
_STRATEGY_RETURN_CAP_NEG_MIN = {"단기": -0.12, "중기": -0.25, "장기": -0.50}

# ✅ 표시 안정화용 파라미터
_MIN_RANGE_WIDTH = _default_config["CLASS_BIN"]["min_width"]
_ROUND_DECIMALS  = 4
_EPS_ZERO_BAND   = _default_config["CLASS_BIN"]["zero_band_eps"]
_DISPLAY_MIN_RET = 1e-4

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

# ------------------------
# config.json 로드/생성
# ------------------------
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
        _config = _loaded if isinstance(_loaded, dict) else _default_config.copy()
        _deep_merge(_config, _default_config)  # 누락키 보강
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
    global _dynamic_num_classes, NUM_CLASSES
    _dynamic_num_classes = int(n)
    try:
        NUM_CLASSES = _dynamic_num_classes
    except Exception:
        pass

def get_NUM_CLASSES():
    global _dynamic_num_classes
    return _dynamic_num_classes if _dynamic_num_classes is not None else _config.get("NUM_CLASSES", _default_config["NUM_CLASSES"])

def get_FEATURE_INPUT_SIZE(): return _config.get("FEATURE_INPUT_SIZE", _default_config["FEATURE_INPUT_SIZE"])
def get_FAIL_AUGMENT_RATIO(): return _config.get("FAIL_AUGMENT_RATIO", _default_config["FAIL_AUGMENT_RATIO"])
def get_MIN_FEATURES():       return _config.get("MIN_FEATURES", _default_config["MIN_FEATURES"])
def get_SYMBOLS():            return _config.get("SYMBOLS", _default_config["SYMBOLS"])

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
    _log(f"[📊 클래스 분포 그룹] 총={num_classes}, 그룹크기={group_size}, 그룹수={len(groups)}")
    return groups

# ------------------------
# 신규 옵션 Getter
# ------------------------
def get_REGIME():   return _config.get("REGIME", _default_config["REGIME"])
def get_CALIB():    return _config.get("CALIB", _default_config["CALIB"])
def get_LOSS():     return _config.get("LOSS", _default_config["LOSS"])
def get_AUG():      return _config.get("AUG", _default_config["AUG"])
def get_EVAL():     return _config.get("EVAL", _default_config["EVAL"])
def get_FAILLEARN():return _config.get("FAILLEARN", _default_config["FAILLEARN"])
def get_QUALITY():  return _config.get("QUALITY", _default_config["QUALITY"])
def get_CLASS_BIN():return _config.get("CLASS_BIN", _default_config["CLASS_BIN"])
def get_TRAIN():    return _config.get("TRAIN", _default_config["TRAIN"])
def get_ENSEMBLE(): return _config.get("ENSEMBLE", _default_config["ENSEMBLE"])
def get_SCHED():    return _config.get("SCHED", _default_config["SCHED"])
def get_PATTERN():  return _config.get("PATTERN", _default_config["PATTERN"])
def get_BLEND():    return _config.get("BLEND", _default_config["BLEND"])
def get_PUBLISH():  return _config.get("PUBLISH", _default_config["PUBLISH"])

# --- 🔧 DATA / CLASS_ENFORCE / CV_CONFIG 런타임 Getter (ENV 오버라이드 지원) ---
def _env_bool(v): return str(v).strip().lower() not in {"0","false","no","off","none",""}

def get_CLASS_ENFORCE() -> dict:
    base = dict(_config.get("CLASS_ENFORCE", _default_config["CLASS_ENFORCE"]))
    ov = os.getenv("CLASS_N_OVERRIDE", None)
    if ov is not None:
        try: base["n_override"] = int(ov)
        except Exception: pass
    s1 = os.getenv("CLASS_SAME_ACROSS_GROUPS", None)
    if s1 is not None: base["same_across_groups"] = _env_bool(s1)
    s2 = os.getenv("CLASS_SAME_ACROSS_SYMBOLS", None)
    if s2 is not None: base["same_across_symbols"] = _env_bool(s2)
    return base

def _data_from_env(base: dict) -> dict:
    d = dict(base or {})
    v = os.getenv("ENABLE_DATA_MERGE", None)
    if v is not None: d["merge_enabled"] = _env_bool(v)
    pv = os.getenv("DATA_PREFER", None)
    if pv is not None: d["prefer"] = str(pv).strip().lower()
    tol = os.getenv("DATA_ALIGN_TOL_SEC", None)
    if tol is not None:
        try:
            d.setdefault("align", {})["tolerance_sec"] = int(tol)
        except Exception:
            pass
    fg = os.getenv("DATA_FILL_MAX_GAP", None)
    if fg is not None:
        try:
            d.setdefault("fill", {})["max_gap"] = int(fg)
        except Exception:
            pass
    return d

def get_DATA() -> dict:
    return _config.get("DATA", _default_config["DATA"])

def get_DATA_RUNTIME() -> dict:
    return _data_from_env(get_DATA())

def get_CV_CONFIG() -> dict:
    base = dict(_config.get("CV_CONFIG", _default_config["CV_CONFIG"]))
    f = os.getenv("CV_FOLDS", None)
    if f is not None:
        try: base["folds"] = int(f)
        except Exception: pass
    mpc = os.getenv("CV_MIN_PER_CLASS", None)
    if mpc is not None:
        try: base["min_per_class"] = int(mpc)
        except Exception: pass
    fr = os.getenv("CV_FALLBACK_REDUCE_FOLDS", None)
    if fr is not None: base["fallback_reduce_folds"] = _env_bool(fr)
    fs = os.getenv("CV_FALLBACK_STRATIFIED", None)
    if fs is not None: base["fallback_stratified"] = _env_bool(fs)
    return base

# --- ONCHAIN Getter (신설) ---
def get_ONCHAIN() -> dict:
    return _config.get("ONCHAIN", _default_config["ONCHAIN"])

# ------------------------
# 헬퍼
# ------------------------
def _ROUNDS(): return _ROUNDS_DECIMALS if 'ROUNDS_DECIMALS' in globals() else _ROUNDS_DECIMALS
_ROUNDS_DECIMALS = 4
def _round2(x: float) -> float: return round(float(x), _ROUNDS_DECIMALS)

def _cap_by_strategy(x: float, strategy: str) -> float:
    pos_cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy)
    neg_cap = _STRATEGY_RETURN_CAP_NEG_MIN.get(strategy)
    if x > 0 and pos_cap is not None: return min(x, pos_cap)
    if x < 0 and neg_cap is not None: return max(x, neg_cap)
    return x

def _enforce_min_width(low: float, high: float):
    if (high - low) < _MIN_RANGE_WIDTH: high = low + _MIN_RANGE_WIDTH
    return low, high

def _fix_monotonic(ranges):
    fixed, prev_hi = [], None
    for lo, hi in ranges:
        if prev_hi is not None and lo < prev_hi:
            lo = prev_hi
            lo, hi = _enforce_min_width(lo, hi)
        lo, hi = _round2(lo), _round2(hi)
        if hi <= lo: hi = _round2(lo + _MIN_RANGE_WIDTH)
        fixed.append((lo, hi))
        prev_hi = hi
    return fixed

def _ensure_zero_band(ranges):
    crosses = [i for i,(lo,hi) in enumerate(ranges) if lo < 0.0 <= hi]
    if crosses:
        i = crosses[0]
        lo, hi = ranges[i]
        if (hi - lo) < max(_MIN_RANGE_WIDTH, _EPS_ZERO_BAND*2):
            lo, hi = min(lo, -_EPS_ZERO_BAND), max(hi, _EPS_ZERO_BAND)
            ranges[i] = (_round2(lo), _round2(hi))
        return ranges
    left_idx  = max([i for i,(lo,hi) in enumerate(ranges) if hi <= 0.0], default=None)
    right_idx = min([i for i,(lo,hi) in enumerate(ranges) if lo >  0.0], default=None)
    if left_idx is None or right_idx is None: return ranges
    lo_l, hi_l = ranges[left_idx]
    lo_r, hi_r = ranges[right_idx]
    ranges[left_idx]  = (_round2(lo_l), _round2(-_EPS_ZERO_BAND))
    ranges[right_idx] = (_round2(_EPS_ZERO_BAND), _round2(hi_r))
    ranges = ranges[:right_idx] + [(_round2(-_EPS_ZERO_BAND), _round2(_EPS_ZERO_BAND))] + ranges[right_idx:]
    return _fix_monotonic(ranges)

def _strictify(ranges):
    if not ranges: return []
    fixed, lo = [], float(ranges[0][0])
    for _, hi in ranges:
        hi = float(hi)
        if hi <= lo: hi = lo + _MIN_RANGE_WIDTH
        lo_r, hi_r = _round2(lo), _round2(hi)
        if hi_r <= lo_r: hi_r = _round2(lo_r + _MIN_RANGE_WIDTH)
        fixed.append((lo_r, hi_r))
        lo = hi_r
    return fixed

def _strategy_horizon_hours(strategy: str) -> int:
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)

def _future_extreme_signed_returns(df, horizon_hours: int):
    import numpy as np, pandas as pd
    if df is None or len(df) == 0 or "timestamp" not in df.columns or "close" not in df.columns:
        return np.zeros(0, dtype=np.float32)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None: ts = ts.dt.tz_localize("Asia/Seoul")
    else: ts = ts.dt.tz_convert("Asia/Seoul")
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().astype(float).values
    high  = pd.to_numeric(df["high"] if "high" in df.columns else df["close"], errors="coerce").ffill().bfill().astype(float).values
    low   = pd.to_numeric(df["low"]  if "low"  in df.columns else df["close"], errors="coerce").ffill().bfill().astype(float).values
    horizon = pd.Timedelta(hours=int(horizon_hours))
    up = np.zeros(len(df), dtype=np.float32); dn = np.zeros(len(df), dtype=np.float32)
    j_up = j_dn = 0
    for i in range(len(df)):
        t1 = ts.iloc[i] + horizon
        j = max(j_up, i); max_h = high[i]
        while j < len(df) and ts.iloc[j] < t1:
            if high[j] > max_h: max_h = high[j]
            j += 1
        j_up = max(j_up, i)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        up[i] = float((max_h - base) / (base + 1e-12))
        k = max(j_dn, i); min_l = low[i]
        while k < len(df) and ts.iloc[k] < t1:
            if low[k] < min_l: min_l = low[k]
            k += 1
        j_dn = max(j_dn, i)
        dn[i] = float((min_l - base) / (base + 1e-12))
    return np.concatenate([dn, up]).astype(np.float32)

# ---- 동적 bin 개수 결정 로직 ---------------------------------------
def _choose_n_classes(rets_signed, max_classes, hint_min=4):
    import numpy as np
    N = int(rets_signed.size)
    if N <= 1: return max(4, hint_min)
    q25, q75 = np.quantile(rets_signed, [0.25, 0.75])
    iqr = float(q75 - q25)
    data_min, data_max = float(np.min(rets_signed)), float(np.max(rets_signed))
    data_range = max(1e-12, data_max - data_min)
    if iqr <= 1e-12:
        est = int(round(np.sqrt(N)))
    else:
        h = 2.0 * iqr * (N ** (-1.0/3.0))
        est = int(round(data_range / max(h, 1e-12)))
    base_hint = int(_config.get("NUM_CLASSES", 10))
    lower = max(4, hint_min, min(base_hint, max_classes) if est < 4 else 4)
    n_cls = max(lower, min(est, max_classes))
    return int(n_cls)

def _merge_smallest_adjacent(ranges, max_classes):
    if not ranges or len(ranges) <= max_classes: return ranges
    import numpy as np
    rs = list(ranges)
    while len(rs) > max_classes:
        widths = np.array([hi - lo for (lo, hi) in rs], dtype=float)
        idx = int(np.argmin(widths))
        if idx == 0: rs[0] = (rs[0][0], rs[1][1]); del rs[1]
        elif idx == len(rs) - 1: rs[-2] = (rs[-2][0], rs[-1][1]); del rs[-1]
        else:
            left_w  = rs[idx][0] - rs[idx-1][0] if idx-1 >= 0 else float("inf")
            right_w = rs[idx+1][1] - rs[idx][1] if idx+1 < len(rs) else float("inf")
            if left_w <= right_w: rs[idx-1] = (rs[idx-1][0], rs[idx][1]); del rs[idx]
            else: rs[idx] = (rs[idx][0], rs[idx+1][1]); del rs[idx+1]
    return rs

def _merge_sparse_bins_by_hist(ranges, rets_signed, max_classes, bin_conf):
    import numpy as np
    if not ranges or rets_signed is None or rets_signed.size == 0: return ranges
    opt = (bin_conf or {}).get("merge_sparse", {})
    env_enabled = os.getenv("MERGE_SPARSE_ENABLED", None)
    if env_enabled is not None:
        opt = dict(opt or {})
        opt["enabled"] = str(env_enabled).strip().lower() not in {"0", "false", "no"}
    if not opt or not opt.get("enabled", False):
        return ranges
    env_ratio = os.getenv("MERGE_SPARSE_MIN_RATIO", None)
    env_floor = os.getenv("MERGE_SPARSE_MIN_FLOOR", None)

    total = int(rets_signed.size)
    min_ratio = float(env_ratio) if env_ratio is not None else float(opt.get("min_ratio", 0.02))
    min_floor = int(env_floor) if env_floor is not None else int(opt.get("min_count_floor", 80))
    prefer = str(opt.get("prefer", "denser")).lower()
    edges = [ranges[0][0]] + [hi for (_, hi) in ranges]
    edges[-1] = float(edges[-1]) + 1e-12
    hist, _ = np.histogram(rets_signed, bins=np.array(edges, dtype=float))
    rs = list(ranges)

    def _rebuild_edges(rr):
        ee = [rr[0][0]] + [hi for (_, hi) in rr]
        ee[-1] = float(ee[-1]) + 1e-12
        return np.array(ee, dtype=float)
    def _counts(rr):
        ee = _rebuild_edges(rr)
        h, _ = np.histogram(rets_signed, bins=ee)
        return h

    changed = True
    while changed:
        changed = False
        if len(rs) <= 2: break
        counts = _counts(rs)
        thresh = max(int(total * min_ratio), min_floor)
        sparse_idxs = [i for i, c in enumerate(counts) if c < thresh]
        if not sparse_idxs: break
        i = int(sorted(sparse_idxs, key=lambda k: counts[k])[0])
        if prefer == "left" and i > 0: j = i - 1
        elif prefer == "right" and i < len(rs) - 1: j = i + 1
        else:
            left_ok, right_ok = i - 1 >= 0, i + 1 < len(rs)
            if left_ok and right_ok: j = i - 1 if counts[i - 1] >= counts[i + 1] else i + 1
            elif left_ok: j = i - 1
            elif right_ok: j = i + 1
            else: break
        lo, hi = min(rs[i][0], rs[j][0]), max(rs[i][1], rs[j][1])
        rs[min(i, j)] = (float(lo), float(hi))
        del rs[max(i, j)]
        changed = True
        if len(rs) > max_classes:
            rs = _merge_smallest_adjacent(rs, max_classes)

    rs = [(float(lo), float(hi)) for (lo, hi) in rs]
    rs = _fix_monotonic(rs)
    rs = _ensure_zero_band(rs)
    if get_CLASS_BIN().get("strict", True):
        rs = _strictify(rs)
    if len(rs) > max_classes:
        rs = _merge_smallest_adjacent(rs, max_classes)
    return rs

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

def get_class_ranges(symbol=None, strategy=None, method=None, group_id=None, group_size=5):
    import numpy as np
    from data.utils import get_kline_by_strategy  # 프로젝트 환경에 맞춘 경로 유지

    MAX_CLASSES = int(_config.get("MAX_CLASSES", _default_config["MAX_CLASSES"]))
    BIN_CONF = get_CLASS_BIN()
    method_req = (os.getenv("CLASS_BIN_METHOD") or method or BIN_CONF.get("method") or "quantile").lower()

    # --- 클래스 수 정책 반영 (일관성/강제 고정) ---
    ce = get_CLASS_ENFORCE()
    n_override = ce.get("n_override", None)
    if n_override is not None:
        try:
            n_override = int(n_override)
            if n_override >= 2:
                set_NUM_CLASSES(n_override)
        except Exception:
            pass

    def compute_equal_ranges(n_cls, reason=""):
        n_cls = max(4, int(n_cls))
        neg = _STRATEGY_RETURN_CAP_NEG_MIN.get(strategy, -0.5)
        pos = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy,  0.5)
        step = (pos - neg) / n_cls
        ranges = []
        for i in range(n_cls):
            lo, hi = neg + i * step, neg + (i + 1) * step
            lo, hi = _enforce_min_width(lo, hi)
            lo, hi = _cap_by_strategy(lo, strategy), _cap_by_strategy(hi, strategy)
            ranges.append((_round2(lo), _round2(hi)))
        if reason: _log(f"[⚠️ 균등 분할 클래스 사용] 사유: {reason}")
        ranges = _fix_monotonic(ranges); ranges = _ensure_zero_band(ranges)
        if BIN_CONF.get("strict", True): ranges = _strictify(ranges)
        if len(ranges) > MAX_CLASSES: ranges = _merge_smallest_adjacent(ranges, MAX_CLASSES)
        return ranges

    def compute_fixed_step_ranges(rets_for_merge):
        env_step = os.getenv("CLASS_BIN_STEP") or os.getenv("DYN_CLASS_STEP")
        step = float(env_step) if env_step is not None else float(BIN_CONF.get("step_pct", 0.0050))
        if step <= 0: step = 0.0050
        neg = _STRATEGY_RETURN_CAP_NEG_MIN.get(strategy, -0.5)
        pos = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy,  0.5)

        edges, val = [], float(neg)
        while val < pos - 1e-12:
            edges.append(val); val += step
        edges.append(pos)
        if len(edges) < 2:
            return compute_equal_ranges(get_NUM_CLASSES(), reason="fixed_step edges 부족")

        cooked = []
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            lo, hi = _cap_by_strategy(lo, strategy), _cap_by_strategy(hi, strategy)
            lo, hi = _enforce_min_width(lo, hi)
            cooked.append((_round2(lo), _round2(hi)))

        fixed = _fix_monotonic(cooked); fixed = _ensure_zero_band(fixed)
        if BIN_CONF.get("strict", True): fixed = _strictify(fixed)
        if rets_for_merge is not None and rets_for_merge.size > 0:
            fixed = _merge_sparse_bins_by_hist(fixed, rets_for_merge, MAX_CLASSES, BIN_CONF)
        if len(fixed) > MAX_CLASSES: fixed = _merge_smallest_adjacent(fixed, MAX_CLASSES)
        if not fixed or len(fixed) < 2:
            return compute_equal_ranges(get_NUM_CLASSES(), reason="fixed_step 최종 경계 부족")
        return fixed

    def compute_ranges_from_kline():
        try:
            df_price = get_kline_by_strategy(symbol, strategy)
            if df_price is None or len(df_price) < 30 or "close" not in df_price:
                return compute_equal_ranges(get_NUM_CLASSES(), reason="가격 데이터 부족")
            horizon_hours = _strategy_horizon_hours(strategy)
            rets_signed = _future_extreme_signed_returns(df_price, horizon_hours=horizon_hours)
            rets_signed = rets_signed[np.isfinite(rets_signed)]
            if rets_signed.size < 10:
                return compute_equal_ranges(get_NUM_CLASSES(), reason="수익률 샘플 부족")
            rets_signed = np.array([_cap_by_strategy(float(r), strategy) for r in rets_signed], dtype=np.float32)

            n_cls = _choose_n_classes(
                rets_signed,
                max_classes=int(_config.get("MAX_CLASSES", 12)),
                hint_min=int(_config.get("NUM_CLASSES", 10))
            )
            # 일관성 강제: n_override가 있으면 사용
            if isinstance(n_override, int) and n_override >= 2:
                n_cls = n_override

            method2 = method_req
            qs = (np.quantile(rets_signed, np.linspace(0, 1, n_cls + 1))
                  if method2 == "quantile"
                  else np.linspace(float(rets_signed.min()), float(rets_signed.max()), n_cls + 1))
            cooked = []
            for i in range(n_cls):
                lo, hi = float(qs[i]), float(qs[i + 1])
                lo, hi = _cap_by_strategy(lo, strategy), _cap_by_strategy(hi, strategy)
                lo, hi = _enforce_min_width(lo, hi)
                cooked.append((_round2(lo), _round2(hi)))
            fixed = _fix_monotonic(cooked); fixed = _ensure_zero_band(fixed)
            if BIN_CONF.get("strict", True): fixed = _strictify(fixed)
            if len(fixed) > int(_config.get("MAX_CLASSES", 12)):
                fixed = _merge_smallest_adjacent(fixed, int(_config.get("MAX_CLASSES", 12)))
            if not fixed or len(fixed) < 2:
                return compute_equal_ranges(get_NUM_CLASSES(), reason="최종 경계 부족(가드)")
            fixed = _merge_sparse_bins_by_hist(fixed, rets_signed, MAX_CLASSES, BIN_CONF)
            return fixed
        except Exception as e:
            return compute_equal_ranges(get_NUM_CLASSES(), reason=f"예외 발생: {e}")

    if method_req == "fixed_step":
        try:
            from data.utils import get_kline_by_strategy as _dbg_k
            df_dbg = _dbg_k(symbol, strategy)
            if df_dbg is not None and len(df_dbg) >= 2 and "close" in df_dbg:
                rets_for_merge = _future_extreme_signed_returns(df_dbg, horizon_hours=_strategy_horizon_hours(strategy))
                rets_for_merge = rets_for_merge[np.isfinite(rets_for_merge)]
            else:
                rets_for_merge = None
        except Exception:
            rets_for_merge = None
        all_ranges = compute_fixed_step_ranges(rets_for_merge)
    else:
        all_ranges = compute_ranges_from_kline()

    if symbol is not None and strategy is not None:
        _ranges_cache[(symbol, strategy)] = all_ranges

    # 디버그 출력(옵션)
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
                    print(f"[📈 수익률분포(±)] {symbol}-{strategy} min={_r2(qs[0])}, p25={_r2(qs[1])}, p50={_r2(qs[2])}, p75={_r2(qs[3])}, p90={_r2(qs[4])}, p95={_r2(qs[5])}, p99={_r2(qs[6])}, max={_r2(qs[7])}")
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

    try:
        if isinstance(all_ranges, list) and len(all_ranges) >= 2:
            set_NUM_CLASSES(len(all_ranges))
    except Exception:
        pass

    if group_id is None: return all_ranges
    start, end = int(group_id) * int(group_size), int(group_id) * int(group_size) + int(group_size)
    if start >= len(all_ranges): return []
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

PREDICT_MIN_RETURN = _get_float("PREDICT_MIN_RETURN", 0.01)
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
# ✅ 순서 1 전역 상수(ENV override 지원)
# ------------------------
_DFLT_STEP = str(_config.get("CLASS_BIN", {}).get("step_pct", 0.0050))
DYN_CLASS_STEP = float(os.getenv("CLASS_BIN_STEP", os.getenv("DYN_CLASS_STEP", _DFLT_STEP)))
BOUNDARY_BAND = float(os.getenv("BOUNDARY_BAND", "0.0020"))
CV_FOLDS   = int(os.getenv("CV_FOLDS", "5"))
CV_GATE_F1 = float(os.getenv("CV_GATE_F1", "0.50"))

# ------------------------
# ✅ 발송(PUBLISH) 섹션 ENV 오버라이드 + 필터 헬퍼
# ------------------------
def _publish_from_env(base: dict) -> dict:
    d = dict(base or {})
    def _b(k, env, cast):
        v = os.getenv(env, None)
        if v is None: return
        try:
            d[k] = cast(v)
        except Exception:
            pass

    _b("enabled", "PUBLISH_ENABLED", lambda x: str(x).strip() not in {"0","false","False"})
    _b("recent_window", "PUBLISH_RECENT_WINDOW", int)
    _b("recent_success_min", "PUBLISH_RECENT_SUCCESS_MIN", float)
    _b("min_expected_return", "PUBLISH_MIN_EXPECTED_RETURN", float)
    _b("abstain_prob_min", "PUBLISH_ABSTAIN_PROB_MIN", float)
    _b("min_meta_confidence", "PUBLISH_MIN_META_CONFIDENCE", float)
    _b("allow_shadow", "PUBLISH_ALLOW_SHADOW", lambda x: str(x).strip() not in {"0","false","False"})
    _b("always_log", "PUBLISH_ALWAYS_LOG", lambda x: str(x).strip() not in {"0","false","False"})
    return d

# 호출 시점마다 최신 ENV 반영
def get_PUBLISH_RUNTIME() -> dict:
    base = get_PUBLISH()
    return _publish_from_env(base)

def passes_publish_filter(*, meta_confidence=None, recent_success_rate=None,
                          expected_return=None, calib_prob=None, shadow=False, source=None):
    """
    발송여부 판단 헬퍼.
    - predict는 항상 기록, 발송은 여기 True일 때만 진행.
    - 반환: (ok:bool, reason:str, thresholds:dict)
    """
    cfg = get_PUBLISH_RUNTIME()
    thr = {
        "enabled": bool(cfg.get("enabled", True)),
        "recent_window": int(cfg.get("recent_window", 10)),
        "recent_success_min": float(cfg.get("recent_success_min", 0.60)),
        "min_expected_return": float(cfg.get("min_expected_return", 0.01)),
        "abstain_prob_min": float(cfg.get("abstain_prob_min", 0.35)),
        "min_meta_confidence": float(cfg.get("min_meta_confidence", 0.0)),
        "allow_shadow": bool(cfg.get("allow_shadow", True)),
        "always_log": bool(cfg.get("always_log", True)),
    }

    if not thr["enabled"]:
        return (False, "publish_disabled", thr)

    if shadow and not thr["allow_shadow"]:
        return (False, "shadow_not_allowed", thr)

    if expected_return is not None and abs(float(expected_return)) < thr["min_expected_return"]:
        return (False, "min_expected_return_not_met", thr)

    if calib_prob is not None and float(calib_prob) < thr["abstain_prob_min"]:
        return (False, "low_confidence", thr)

    if meta_confidence is not None and float(meta_confidence) < thr["min_meta_confidence"]:
        return (False, "low_meta_confidence", thr)

    if recent_success_rate is not None and float(recent_success_rate) < thr["recent_success_min"]:
        return (False, "recent_success_rate_too_low", thr)

    return (True, "ok", thr)

# ------------------------
# ✅ [EVAL_RUNTIME] ENV 오버라이드 + 평가 헬퍼
# ------------------------
def _eval_from_env(base: dict) -> dict:
    d = dict(base or {})
    def _b(k, env, cast):
        v = os.getenv(env, None)
        if v is None: return
        try: d[k] = cast(v)
        except Exception: pass

    _b("timebase", "EVAL_TIMEBASE", str)
    _b("check_interval_min", "EVAL_CHECK_INTERVAL_MIN", int)
    _b("grace_min", "EVAL_GRACE_MIN", int)
    _b("price_window_slack_min", "EVAL_PRICE_WINDOW_SLACK_MIN", int)
    _b("max_backfill_hours", "EVAL_MAX_BACKFILL_HOURS", int)
    return d

def get_EVAL_RUNTIME() -> dict:
    base = _config.get("EVAL_RUNTIME", _default_config["EVAL_RUNTIME"])
    return _eval_from_env(base)

def strategy_horizon_hours(strategy: str) -> int:
    """평가/만기 산정에 쓰는 전략별 horizon (시간)."""
    return _strategy_horizon_hours(strategy)

def compute_eval_due_at(now_utc, strategy: str):
    """predict 시점(now_utc) 기준, 전략별 평가 예정 시각을 반환."""
    from datetime import timedelta
    return now_utc + timedelta(hours=_strategy_horizon_hours(strategy))

# ------------------------
# 전역 캐시된 값(기존)
# ------------------------
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
NUM_CLASSES        = get_NUM_CLASSES()
FAIL_AUGMENT_RATIO = get_FAIL_AUGMENT_RATIO()
MIN_FEATURES       = get_MIN_FEATURES()

# === 캘리브레이션 모듈 호환용 전역 노출 ===
CALIB = get_CALIB()

__all__ = [
    "STRATEGY_CONFIG",
    "get_NUM_CLASSES", "set_NUM_CLASSES",
    "get_FEATURE_INPUT_SIZE",
    "get_class_groups", "get_class_ranges",
    "get_class_return_range", "class_to_expected_return",
    "get_SYMBOLS", "get_SYMBOL_GROUPS",
    "get_REGIME", "get_CALIB", "get_LOSS", "get_AUG", "get_EVAL",
    "get_FAILLEARN", "get_QUALITY",
    "get_CLASS_BIN", "get_TRAIN", "get_ENSEMBLE", "get_SCHED",
    "get_PATTERN", "get_BLEND", "get_PUBLISH", "get_PUBLISH_RUNTIME",
    "passes_publish_filter",
    "get_CPU_THREADS", "get_TRAIN_NUM_WORKERS", "get_TRAIN_BATCH_SIZE",
    "get_ORDERED_TRAIN", "get_PREDICT_MIN_RETURN", "get_DISPLAY_MIN_RETURN",
    "get_SSL_CACHE_DIR",
    "FEATURE_INPUT_SIZE", "NUM_CLASSES", "FAIL_AUGMENT_RATIO", "MIN_FEATURES",
    "CALIB",
    "DYN_CLASS_STEP", "BOUNDARY_BAND", "CV_FOLDS", "CV_GATE_F1",
    "get_EVAL_RUNTIME", "strategy_horizon_hours", "compute_eval_due_at",
    # ▼ 신규 노출
    "get_DATA", "get_DATA_RUNTIME", "get_CLASS_ENFORCE", "get_CV_CONFIG",
    "get_ONCHAIN",
        ]
