# integrity_guard.py
import importlib, os, json, datetime

REQUIRED = {
    "predict": ["predict", "evaluate_predictions", "get_model_predictions"],
    "logger": ["log_prediction", "update_model_success"],  # 이제 두 개만 필요
    "data.utils": ["get_kline_by_strategy", "compute_features"],
    "model.base_model": ["get_model"],
    "model_weight_loader": ["load_model_cached"],
    "config": ["get_NUM_CLASSES", "get_FEATURE_INPUT_SIZE", "get_class_return_range", "class_to_expected_return"],
}

VERSION_FILE = "VERSION.json"

def run():
    problems = []
    for mod, attrs in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            problems.append(f"[모듈없음] {mod} → {e}")
            continue
        for a in attrs:
            if not hasattr(m, a):
                problems.append(f"[심볼없음] {mod}.{a}")
    # VERSION 태그(선택): predict.py가 최신인지 간단히 표시
    stamp = {"last_boot": datetime.datetime.utcnow().isoformat()+"Z", "predict_tag": "self-contained"}
    try:
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            json.dump(stamp, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if problems:
        print("=== [무결성 점검 결과] ===")
        for p in problems:
            print(" -", p)
        print("=== 위 항목을 우선 수정하세요 ===")
    else:
        print("[✅ 무결성 가드 통과] 필수 심볼/모듈 이상 없음")
