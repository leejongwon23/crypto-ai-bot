import json
import os

CONFIG_PATH = "/persistent/config.json"

# ✅ 기본 설정값
_default_config = {
    "NUM_CLASSES": 21,
    "FEATURE_INPUT_SIZE": 21,
    "FAIL_AUGMENT_RATIO": 3
}

# ✅ config.json 로드
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = json.load(f)
        print("[✅ config.py] config.json 로드 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 로드 실패 → 기본값 사용: {e}")
        _config = _default_config.copy()
else:
    _config = _default_config.copy()

# ✅ 저장 함수
def save_config():
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_config, f, ensure_ascii=False, indent=2)
        print("[✅ config.py] config.json 저장 완료")
    except Exception as e:
        print(f"[⚠️ config.py] config.json 저장 실패 → {e}")

# ✅ get 함수들
def get_NUM_CLASSES():
    return _config.get("NUM_CLASSES", _default_config["NUM_CLASSES"])

def get_FEATURE_INPUT_SIZE():
    return _config.get("FEATURE_INPUT_SIZE", _default_config["FEATURE_INPUT_SIZE"])

def get_FAIL_AUGMENT_RATIO():
    return _config.get("FAIL_AUGMENT_RATIO", _default_config["FAIL_AUGMENT_RATIO"])

# ✅ set 함수들
def set_NUM_CLASSES(value):
    _config["NUM_CLASSES"] = int(value)
    save_config()

def set_FEATURE_INPUT_SIZE(value):
    _config["FEATURE_INPUT_SIZE"] = int(value)
    save_config()

def set_FAIL_AUGMENT_RATIO(value):
    _config["FAIL_AUGMENT_RATIO"] = int(value)
    save_config()

# ✅ 클래스 그룹화 함수 (기존 유지)
def get_class_groups(num_classes=None, group_size=7):
    """
    ✅ 클래스 그룹화 함수 (YOPO v4.1)
    - num_classes를 group_size 크기로 나누어 그룹화
    - num_classes ≤ group_size 시 단일 그룹 반환
    """
    if num_classes is None:
        num_classes = get_NUM_CLASSES()
    if num_classes <= group_size:
        return [list(range(num_classes))]
    return [list(range(i, min(i+group_size, num_classes))) for i in range(0, num_classes, group_size)]

def get_class_ranges():
    num_classes = get_NUM_CLASSES()
    # ✅ 균등 분할 예시 (-1.0 ~ +1.0 범위)
    step = 2.0 / num_classes
    ranges = [(-1.0 + i*step, -1.0 + (i+1)*step) for i in range(num_classes)]
    return ranges
