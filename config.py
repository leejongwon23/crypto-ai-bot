import json
import os

CONFIG_PATH = "/persistent/config.json"

# ✅ 기본 설정값
_default_config = {
    "NUM_CLASSES": 20,
    "FEATURE_INPUT_SIZE": 24,
    "FAIL_AUGMENT_RATIO": 3,
    "MIN_FEATURES": 5,
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"],
    "SYMBOL_GROUP_SIZE": 3,
}

# ✅ 거래 전략별 interval + limit 설정 (Bybit/Binance 호환)
STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 1000, "binance_interval": "4h"},
    "중기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
    "장기": {"interval": "D",   "limit": 500,  "binance_interval": "1d"},
}

# ✅ 내부 동적 캐시 변수
_config = _default_config.copy()
_dynamic_num_classes = None

# (symbol, strategy) → ranges(list[(min,max)])
_ranges_cache = {}   # 예: {("BTCUSDT","단기"): [(-0.05,-0.04), ..., (0.04,0.05)]}

# ✅ config.json 로드
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = json.load(f)
        print("[✅ config.py] config.json 로드 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 로드 실패 → 기본값 사용: {e}")

# ✅ 저장 함수
def save_config():
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_config, f, ensure_ascii=False, indent=2)
        print("[✅ config.py] config.json 저장 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 저장 실패 → {e}")

# =========================
# 기본 Getter/Setter
# =========================
def set_NUM_CLASSES(n):
    """동적으로 결정된 클래스 수를 설정 (전역)"""
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

# ✅ 클래스 그룹화
def get_class_groups(num_classes=None, group_size=5):
    if num_classes is None or num_classes < 2:
        num_classes = get_NUM_CLASSES()
    if num_classes <= group_size:
        groups = [list(range(num_classes))]
    else:
        groups = [list(range(i, min(i + group_size, num_classes))) for i in range(0, num_classes, group_size)]

    # 📊 로깅
    print(f"[📊 클래스 그룹화] 총 클래스 수: {num_classes}, 그룹 크기: {group_size}, 그룹 개수: {len(groups)}")
    for gi, g in enumerate(groups):
        print(f"  - 그룹 {gi}: 클래스 {g}")

    return groups

# =========================
# 클래스 경계 계산/조회 (심볼·전략별 일관성 보장)
# =========================
def get_class_return_range(class_id: int, symbol: str, strategy: str):
    """
    심볼/전략별로 계산된 동적 클래스 경계를 사용해 (cls_min, cls_max)를 반환.
    - 학습에서 사용한 경계(get_class_ranges(symbol,strategy))와 동일한 소스 사용
    """
    assert isinstance(symbol, str) and isinstance(strategy, str), "symbol/strategy는 문자열이어야 함"
    num_classes = get_NUM_CLASSES()
    key = (symbol, strategy)

    ranges = _ranges_cache.get(key)
    if ranges is None or len(ranges) != num_classes:
        # 캐시에 없거나 클래스 수가 달라졌으면 재계산
        ranges = get_class_ranges(symbol=symbol, strategy=strategy)
        _ranges_cache[key] = ranges

    assert 0 <= class_id < len(ranges), f"class_id {class_id} 범위 오류 (0~{len(ranges)-1})"
    return ranges[class_id]

def class_to_expected_return(class_id: int, symbol: str, strategy: str):
    """
    해당 클래스의 대표 기대 수익률(중앙값)을 반환.
    """
    r_min, r_max = get_class_return_range(class_id, symbol, strategy)
    return (r_min + r_max) / 2

def get_class_ranges(symbol=None, strategy=None, method="quantile", group_id=None, group_size=5):
    """
    심볼/전략별 수익률 분포 기반 동적 클래스 경계 계산.
    - 음수/양수 구간을 분리한 상태로 클래스 경계를 산출
    - 실패/부족 시 균등 분할 fallback
    - 계산 결과는 (symbol,strategy) 캐시에 저장되어 예측/평가 시 동일하게 사용
    """
    import numpy as np
    from data.utils import get_kline_by_strategy
    from config import set_NUM_CLASSES

    MAX_CLASSES = 20
    MIN_HALF = 2

    def compute_split_ranges_from_kline():
        try:
            df_price = get_kline_by_strategy(symbol, strategy)
            if df_price is None or len(df_price) < 30:
                print(f"[⚠️ get_class_ranges] 가격 데이터 부족 ({len(df_price) if df_price is not None else 0}봉) → fallback equal 사용")
                return compute_equal_ranges(10, reason="가격 데이터 부족")

            returns = df_price["close"].pct_change().dropna().values
            if len(returns) < 10:
                print(f"[⚠️ get_class_ranges] 수익률 데이터 부족 ({len(returns)}) → fallback equal 사용")
                return compute_equal_ranges(10, reason="수익률 부족")

            neg = returns[returns < 0]
            pos = returns[returns >= 0]

            # 음/양 수익률 별 클래스 수 결정
            half_neg = max(MIN_HALF, min(10, len(neg) // 5))
            half_pos = max(MIN_HALF, min(10, len(pos) // 5))

            num_classes = min(MAX_CLASSES, half_neg + half_pos)
            if num_classes % 2 != 0:
                num_classes -= 1
            num_classes = max(num_classes, 4)

            set_NUM_CLASSES(num_classes)

            print(f"[📊 수익률 분포 계산] {symbol}-{strategy}")
            print(f"  - 음수 수익률: {len(neg)}개, 양수 수익률: {len(pos)}개")
            print(f"  - 음수 클래스 수: {num_classes // 2}, 양수 클래스 수: {num_classes // 2}")
            print(f"  - 총 클래스 수: {num_classes} (MAX={MAX_CLASSES})")

            if method == "quantile":
                q_neg = np.quantile(neg, np.linspace(0, 1, num_classes // 2 + 1))
                q_pos = np.quantile(pos, np.linspace(0, 1, num_classes // 2 + 1))
            else:
                q_neg = np.linspace(neg.min(), neg.max(), num_classes // 2 + 1)
                q_pos = np.linspace(pos.min(), pos.max(), num_classes // 2 + 1)

            neg_ranges = [(float(q_neg[i]), float(q_neg[i + 1])) for i in range(num_classes // 2)]
            pos_ranges = [(float(q_pos[i]), float(q_pos[i + 1])) for i in range(num_classes // 2)]

            # ✅ 클래스 경계 로그 출력
            print("  [🔍 손실 구간 클래스]")
            for i, r in enumerate(neg_ranges):
                print(f"    - Class {i}: {r[0]*100:.2f}% ~ {r[1]*100:.2f}%")

            print("  [🔍 수익 구간 클래스]")
            for i, r in enumerate(pos_ranges):
                print(f"    - Class {i + num_classes//2}: {r[0]*100:.2f}% ~ {r[1]*100:.2f}%")

            return neg_ranges + pos_ranges

        except Exception as e:
            print(f"[❌ get_class_ranges] 수익률 계산 예외 발생 → fallback equal 사용: {e}")
            return compute_equal_ranges(10, reason="예외 발생")

    def compute_equal_ranges(n_cls, reason=""):
        # 양/음 구분 없이 균등 분할 (설계 유지)
        step = 2.0 / n_cls
        ranges = [(-1.0 + i * step, -1.0 + (i + 1) * step) for i in range(n_cls)]
        print(f"[⚠️ 균등 분할 클래스 사용] 사유: {reason} → {n_cls}개 클래스, 범위 예시: {ranges[:2]}...")
        return ranges

    all_ranges = compute_split_ranges_from_kline()

    # ✅ 심볼·전략별 캐시에 저장
    if symbol is not None and strategy is not None:
        _ranges_cache[(symbol, strategy)] = all_ranges

    if group_id is None:
        return all_ranges

    start = group_id * group_size
    end = start + group_size
    return all_ranges[start:end]

# ✅ 즉시 변수 선언 (외부 모듈 호환)
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
NUM_CLASSES = get_NUM_CLASSES()
FAIL_AUGMENT_RATIO = get_FAIL_AUGMENT_RATIO()
MIN_FEATURES = get_MIN_FEATURES()
