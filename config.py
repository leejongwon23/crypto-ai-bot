import json
import os

CONFIG_PATH = "/persistent/config.json"

# ✅ 기본 설정값
_default_config = {
    "NUM_CLASSES": 20,               # 전역 기본값(최소 보정용)
    "FEATURE_INPUT_SIZE": 24,
    "FAIL_AUGMENT_RATIO": 3,
    "MIN_FEATURES": 5,
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"],
    "SYMBOL_GROUP_SIZE": 3,
}

# ✅ 전략별 K라인 설정
STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 1000, "binance_interval": "4h"},
    "중기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
    "장기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
}

# ✅ 전략별 양의 수익률 상한(과장 방지용 캡)
#    - 단기: +12%, 중기: +25%, 장기: +50%
_STRATEGY_RETURN_CAP_POS_MAX = {
    "단기": 0.12,
    "중기": 0.25,
    "장기": 0.50,
}

# ✅ 최소 구간 폭 및 반올림 자릿수
_MIN_RANGE_WIDTH = 0.001   # 0.1%
_ROUND_DECIMALS = 3        # 소수 셋째 자리

_config = _default_config.copy()
_dynamic_num_classes = None
_ranges_cache = {}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = json.load(f)
        print("[✅ config.py] config.json 로드 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 로드 실패 → 기본값 사용: {e}")

def save_config():
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_config, f, ensure_ascii=False, indent=2)
        print("[✅ config.py] config.json 저장 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 저장 실패 → {e}")

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

def _round2(x: float) -> float:
    """소수 셋째 자리 반올림(노이즈 제거)."""
    return round(float(x), _ROUND_DECIMALS)

def _cap_positive_by_strategy(x: float, strategy: str) -> float:
    cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy, None)
    if cap is not None and x > 0:
        return min(x, cap)
    return x

def _enforce_min_width(low: float, high: float) -> tuple[float, float]:
    if (high - low) < _MIN_RANGE_WIDTH:
        high = low + _MIN_RANGE_WIDTH
    return low, high

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
    가격 변화율(일반 수익률) 기반으로 음/양 영역을 분할하고,
    - 전략별 양수 캡 적용(과장 방지)
    - 최소 구간 폭 보장(0.1%)
    - 모든 경계 소수 셋째 자리 반올림
    - 경계 단조성/겹침 자동 보정
    """
    import numpy as np
    from data.utils import get_kline_by_strategy

    MAX_CLASSES = 20
    MIN_HALF = 2

    def compute_equal_ranges(n_cls, reason=""):
        step = 2.0 / n_cls  # [-1.0, +1.0] 균등
        raw = [(-1.0 + i * step, -1.0 + (i + 1) * step) for i in range(n_cls)]
        ranges = []
        for lo, hi in raw:
            lo, hi = _enforce_min_width(lo, hi)
            ranges.append((_round2(lo), _round2(hi)))
        print(f"[⚠️ 균등 분할 클래스 사용] 사유: {reason}")
        return _fix_monotonic(ranges)

    def _fix_monotonic(ranges):
        """겹침 제거 및 단조 증가 보정."""
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
        return fixed

    def compute_split_ranges_from_kline():
        try:
            df_price = get_kline_by_strategy(symbol, strategy)
            if df_price is None or len(df_price) < 30:
                return compute_equal_ranges(10, reason="가격 데이터 부족")

            returns = df_price["close"].pct_change().dropna().values
            if len(returns) < 10:
                return compute_equal_ranges(10, reason="수익률 샘플 부족")

            neg = returns[returns < 0]
            pos = returns[returns >= 0]

            # 양수 영역 캡 적용(과장 방지)
            if pos.size > 0 and strategy is not None:
                cap = _STRATEGY_RETURN_CAP_POS_MAX.get(strategy)
                if cap is not None:
                    pos = np.clip(pos, None, cap)

            half_neg = max(MIN_HALF, min(8, len(neg) // 5))
            half_pos = max(MIN_HALF, min(8, len(pos) // 5))

            num_classes = min(MAX_CLASSES, half_neg + half_pos)
            if num_classes % 2 != 0:
                num_classes -= 1
            num_classes = max(num_classes, 4)

            if half_neg + half_pos <= 0:
                return compute_equal_ranges(10, reason="분할 불가")

            # 분위/균등 선택
            if method == "quantile":
                q_neg = np.quantile(neg, np.linspace(0, 1, max(2, half_neg) + 1)) if neg.size > 0 else np.array([-0.05, 0.0])
                q_pos = np.quantile(pos, np.linspace(0, 1, max(2, half_pos) + 1)) if pos.size > 0 else np.array([0.0, 0.05])
            else:
                q_neg = np.linspace(neg.min(), neg.max(), max(2, half_neg) + 1) if neg.size > 0 else np.array([-0.05, 0.0])
                q_pos = np.linspace(pos.min(), pos.max(), max(2, half_pos) + 1) if pos.size > 0 else np.array([0.0, 0.05])

            # 구간 생성
            neg_ranges = [(float(q_neg[i]), float(q_neg[i + 1])) for i in range(max(1, len(q_neg) - 1))]
            pos_ranges = [(float(q_pos[i]), float(q_pos[i + 1])) for i in range(max(1, len(q_pos) - 1))]

            # 최소 폭/반올림/캡 재적용
            cooked = []
            for lo, hi in neg_ranges + pos_ranges:
                lo, hi = _enforce_min_width(lo, hi)
                lo = _cap_positive_by_strategy(lo, strategy) if lo > 0 else lo
                hi = _cap_positive_by_strategy(hi, strategy) if hi > 0 else hi
                lo, hi = _round2(lo), _round2(hi)
                if hi <= lo:
                    hi = _round2(lo + _MIN_RANGE_WIDTH)
                cooked.append((lo, hi))

            return _fix_monotonic(cooked)

        except Exception as e:
            return compute_equal_ranges(10, reason=f"예외 발생: {e}")

    all_ranges = compute_split_ranges_from_kline()

    # 캐시 저장
    if symbol is not None and strategy is not None:
        _ranges_cache[(symbol, strategy)] = all_ranges

    # 그룹 단위 슬라이싱(기존 인터페이스 유지)
    if group_id is None:
        return all_ranges
    return all_ranges[group_id * group_size: (group_id + 1) * group_size]

FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
NUM_CLASSES = get_NUM_CLASSES()
FAIL_AUGMENT_RATIO = get_FAIL_AUGMENT_RATIO()
MIN_FEATURES = get_MIN_FEATURES()
