# integrity_guard.py (YOPO v1.5 — 무결성/예측객체 검증 유틸 포함)
import importlib, os, json, datetime, math
from typing import Any, Tuple, Dict

__all__ = ["run", "ensure_non_nan", "ensure_range_order"]

# 필수 심볼 존재 검사
REQUIRED = {
    "predict": ["predict", "evaluate_predictions", "get_model_predictions"],
    "logger": ["log_prediction", "update_model_success"],
    "data.utils": ["get_kline_by_strategy", "compute_features"],
    "model.base_model": ["get_model"],
    "model_weight_loader": ["load_model_cached"],
    "config": ["get_NUM_CLASSES", "get_FEATURE_INPUT_SIZE",
               "get_class_return_range", "class_to_expected_return"],
    "meta_learning": ["get_meta_prediction", "meta_predict", "select"],  # ← ③ 반영
}

VERSION_FILE = "VERSION.json"

def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False

def ensure_range_order(lo: Any, hi: Any) -> Tuple[bool, float, float, str]:
    """
    lo<=hi 보장. 위반 시 교정 또는 실패 반환.
    반환: (ok, lo_fixed, hi_fixed, reason)
    """
    if not (_is_finite(lo) and _is_finite(hi)):
        return False, 0.0, 0.0, "range_nan"
    lo_f, hi_f = float(lo), float(hi)
    if hi_f < lo_f:
        # swap
        lo_f, hi_f = hi_f, lo_f
        return True, lo_f, hi_f, "range_swapped"
    if hi_f == lo_f:
        try:
            from config import get_CLASS_BIN
            eps = float(get_CLASS_BIN().get("min_width", 0.001))
        except Exception:
            eps = 0.001
        hi_f = lo_f + eps
        return True, lo_f, hi_f, "range_widened"
    return True, lo_f, hi_f, "ok"

def ensure_non_nan(pred: Dict[str, Any], *, ok_if_missing: bool = False) -> Tuple[bool, str]:
    """
    필수 필드 점검:
      - expected_return_mid, class_range_lo, class_range_hi, position, calib_prob
    하나라도 NaN/누락이면 (False, 'integrity_nan') 반환.
    범위 교정은 내부에서 수행(교정 시에도 True 반환).
    """
    keys = ["expected_return_mid", "class_range_lo", "class_range_hi", "position", "calib_prob"]
    for k in keys:
        if k not in pred:
            if ok_if_missing:
                continue
            return False, "integrity_nan"
    # 수치 필드 체크
    if not _is_finite(pred.get("expected_return_mid")):
        return False, "integrity_nan"
    if not _is_finite(pred.get("calib_prob")):
        return False, "integrity_nan"

    ok, lo, hi, why = ensure_range_order(pred.get("class_range_lo"), pred.get("class_range_hi"))
    if not ok:
        return False, "integrity_nan"
    pred["class_range_lo"], pred["class_range_hi"] = lo, hi

    pos = str(pred.get("position", "neutral")).lower()
    if pos not in ("long", "short", "neutral"):
        pred["position"] = "neutral"  # 기본값
    return True, ("ok" if why == "ok" else why)

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
    # VERSION 태그
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

if __name__ == "__main__":
    run()
