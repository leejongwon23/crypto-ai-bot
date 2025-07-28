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
    "SYMBOL_GROUP_SIZE": 3
}

# ✅ 내부 동적 캐시 변수
_config = _default_config.copy()
_dynamic_num_classes = None  # ✅ 동적 클래스 저장

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

# ✅ 클래스별 수익률 범위 반환 (예: 21 클래스 기준)
def get_class_return_range(class_id):
    num_classes = get_NUM_CLASSES()
    assert 0 <= class_id < num_classes, f"class_id {class_id} 잘못됨"
    ranges = get_class_ranges()
    return ranges[class_id]

# ✅ 클래스별 기대 수익률 계산 (중간값 사용)
def class_to_expected_return(class_id, num_classes=None):
    if num_classes is None:
        num_classes = get_NUM_CLASSES()
    r_min, r_max = get_class_return_range(class_id)
    return (r_min + r_max) / 2


# ✅ 동적 클래스 수 설정 함수
def set_NUM_CLASSES(n):
    global _dynamic_num_classes
    _dynamic_num_classes = n

# ✅ get 함수들
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

# ✅ 클래스 그룹화 함수 (클래스 번호 나누기)
def get_class_groups(num_classes=None, group_size=5):
    if num_classes is None:
        num_classes = get_NUM_CLASSES()
    if num_classes <= group_size:
        return [list(range(num_classes))]
    return [list(range(i, min(i+group_size, num_classes))) for i in range(0, num_classes, group_size)]

def get_class_ranges(symbol=None, strategy=None, method="quantile", group_id=None, group_size=5):
    import numpy as np
    from data.utils import get_kline_by_strategy

    num_classes = get_NUM_CLASSES()
    assert num_classes % 2 == 0, "클래스 수는 짝수여야 양/음 분할 가능"

    def compute_split_ranges_from_kline():
        try:
            df_price = get_kline_by_strategy(symbol, strategy)
            if df_price is None or len(df_price) < num_classes:
                print(f"[⚠️ get_class_ranges] 가격 데이터 부족 → fallback equal 사용")
                return compute_equal_ranges()

            returns = df_price["close"].pct_change().dropna().values
            if len(returns) < num_classes:
                print(f"[⚠️ get_class_ranges] 수익률 부족 → fallback equal 사용")
                return compute_equal_ranges()

            half = num_classes // 2
            neg = returns[returns < 0]
            pos = returns[returns >= 0]

            if len(neg) < half or len(pos) < half:
                print(f"[⚠️ get_class_ranges] 양/음 수익률 분포 불충분 → fallback equal 사용")
                return compute_equal_ranges()

            if method == "quantile":
                q_neg = np.quantile(neg, np.linspace(0, 1, half + 1))
                q_pos = np.quantile(pos, np.linspace(0, 1, half + 1))
            else:
                q_neg = np.linspace(neg.min(), neg.max(), half + 1)
                q_pos = np.linspace(pos.min(), pos.max(), half + 1)

            neg_ranges = [(q_neg[i], q_neg[i+1]) for i in range(half)]
            pos_ranges = [(q_pos[i], q_pos[i+1]) for i in range(half)]

            return neg_ranges + pos_ranges

        except Exception as e:
            print(f"[❌ get_class_ranges] 수익률 분포 계산 실패 → fallback equal 사용: {e}")
            return compute_equal_ranges()

    def compute_equal_ranges():
        step = 2.0 / num_classes
        return [(-1.0 + i * step, -1.0 + (i + 1) * step) for i in range(num_classes)]

    # ✅ 실제 수익률 기반으로 계산
    all_ranges = compute_split_ranges_from_kline()

    if group_id is None:
        return all_ranges

    start = group_id * group_size
    end = start + group_size
    return all_ranges[start:end]

# ✅ 즉시 변수 선언
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
NUM_CLASSES = get_NUM_CLASSES()
FAIL_AUGMENT_RATIO = get_FAIL_AUGMENT_RATIO()
MIN_FEATURES = get_MIN_FEATURES()
