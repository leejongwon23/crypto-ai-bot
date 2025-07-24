import json
import os

CONFIG_PATH = "/persistent/config.json"

# ✅ 기본 설정값
_default_config = {
    "NUM_CLASSES": 21,
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

# ✅ 클래스 그룹화 함수
def get_class_groups(num_classes=None, group_size=5):
    if num_classes is None:
        num_classes = get_NUM_CLASSES()
    if num_classes <= group_size:
        return [list(range(num_classes))]
    return [list(range(i, min(i+group_size, num_classes))) for i in range(0, num_classes, group_size)]

# ✅ 클래스 구간 계산 함수
def get_class_ranges(method="equal", data_path="/persistent/prediction_log.csv"):
    import pandas as pd
    import numpy as np

    num_classes = get_NUM_CLASSES()

    if method == "equal":
        step = 2.0 / num_classes
        ranges = [(-1.0 + i*step, -1.0 + (i+1)*step) for i in range(num_classes)]
        return ranges

    elif method == "quantile":
        try:
            df = pd.read_csv(data_path, encoding="utf-8-sig")
            returns = df["return"].dropna().values
            if len(returns) < num_classes:
                print(f"[⚠️ get_class_ranges] 데이터 부족 → equal binning 사용")
                return get_class_ranges(method="equal")

            quantiles = np.quantile(returns, np.linspace(0, 1, num_classes + 1))
            ranges = [(quantiles[i], quantiles[i+1]) for i in range(num_classes)]
            return ranges

        except Exception as e:
            print(f"[❌ get_class_ranges] quantile binning 실패 → {e}")
            return get_class_ranges(method="equal")

    else:
        print(f"[⚠️ get_class_ranges] 알 수 없는 method={method} → equal 사용")
        return get_class_ranges(method="equal")

# ✅ 즉시 변수 선언
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
NUM_CLASSES = get_NUM_CLASSES()
FAIL_AUGMENT_RATIO = get_FAIL_AUGMENT_RATIO()
MIN_FEATURES = get_MIN_FEATURES()
