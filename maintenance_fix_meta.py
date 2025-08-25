# maintenance_fix_meta.py (FINAL, 12번+확장자 보정: calibration/regime_cfg 점검·복구 + saved_at↔timestamp 호환 + weight 확장자 동기화)
import os
import json
import re
import datetime
import pytz
import torch
import glob

from model.base_model import get_model  # 현재 모델 구조 확인용
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE

NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

MODEL_DIR = "/persistent/models"
os.makedirs(MODEL_DIR, exist_ok=True)  # ← 안전 보강: 디렉터리 보장
_KNOWN_EXTS = (".ptz", ".safetensors", ".pt")  # 선호 순서(압축 우선)

KST = pytz.timezone("Asia/Seoul")
now_kst = lambda: datetime.datetime.now(KST).isoformat()

# 파일명 패턴 예:
#  BTCUSDT_단기_lstm_group1_cls3.meta.json
FILENAME_RE = re.compile(
    r"^(?P<symbol>[^_]+)_(?P<strategy>[^_]+)_(?P<model>[^_]+)"
    r"(?:_group(?P<group_id>\d+))?"
    r"(?:_cls(?P<num_classes>\d+))?$"
)

REQUIRED_TOP_FIELDS = [
    "symbol", "strategy", "model", "group_id",
    "input_size", "num_classes", "class_bins",
    "metrics", "saved_at"
]

_ALLOWED_STRATEGIES = {"단기", "중기", "장기"}
_ALLOWED_MODELS = {"lstm", "cnn_lstm", "transformer"}

def _stem_from_meta_filename(meta_filename: str) -> str:
    # "xxx.meta.json" -> "xxx"
    return meta_filename[:-10] if meta_filename.endswith(".meta.json") else os.path.splitext(meta_filename)[0]

def _parse_from_filename(fname: str):
    base = _stem_from_meta_filename(fname)
    m = FILENAME_RE.match(base)
    if not m:
        return {}
    gd = m.groupdict()
    out = {
        "symbol": gd.get("symbol"),
        "strategy": gd.get("strategy"),
        "model": gd.get("model"),
        "group_id": int(gd["group_id"]) if gd.get("group_id") is not None else 0,
        "num_classes": int(gd["num_classes"]) if gd.get("num_classes") is not None else NUM_CLASSES,
    }
    return out

def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("meta json is not an object")
            return data, None
    except Exception as e:
        return None, e

def _safe_write_json(path: str, obj: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[❌ 저장 실패] {os.path.basename(path)} → {e}")
        return False

def _ensure_metrics(meta: dict):
    metrics = meta.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    if "val_acc" not in metrics or not isinstance(metrics.get("val_acc"), (int, float)):
        metrics["val_acc"] = 0.0
    if "val_f1" not in metrics or not isinstance(metrics.get("val_f1"), (int, float)):
        metrics["val_f1"] = 0.0
    if "train_loss_sum" not in metrics or not isinstance(metrics.get("train_loss_sum"), (int, float)):
        metrics["train_loss_sum"] = 0.0
    meta["metrics"] = metrics

def _ensure_calibration(meta: dict):
    cal = meta.get("calibration")
    if not isinstance(cal, dict):
        cal = {}
    method = cal.get("method", "none")
    if method not in ["none", "temperature", "platt"]:
        method = "none"
    # temperature
    try:
        temperature = float(cal.get("temperature", 1.0))
        if temperature <= 0 or not (temperature == temperature):
            temperature = 1.0
    except Exception:
        temperature = 1.0
    # platt
    platt = cal.get("platt")
    if not isinstance(platt, dict):
        platt = {}
    try:
        a = float(platt.get("a", 0.0))
    except Exception:
        a = 0.0
    try:
        b = float(platt.get("b", 0.0))
    except Exception:
        b = 0.0
    try:
        ver = int(cal.get("ver", 1))
    except Exception:
        ver = 1
    updated_at = cal.get("updated_at") or now_kst()
    meta["calibration"] = {
        "method": method,
        "temperature": float(temperature),
        "platt": {"a": float(a), "b": float(b)},
        "updated_at": updated_at,
        "ver": int(ver)
    }

def _ensure_regime_cfg(meta: dict):
    rc = meta.get("regime_cfg")
    if not isinstance(rc, dict):
        rc = {}
    enabled = bool(rc.get("enabled", False))
    detector = rc.get("detector", "none")
    if detector not in ["none", "simple", "volatility"]:
        detector = "none"
    params = rc.get("params")
    if not isinstance(params, dict):
        params = {}
    try:
        ver = int(rc.get("ver", 1))
    except Exception:
        ver = 1
    meta["regime_cfg"] = {
        "enabled": bool(enabled),
        "detector": detector,
        "params": params,
        "ver": int(ver)
    }

def _ensure_saved_ts_fields(meta: dict):
    saved_at = meta.get("saved_at")
    timestamp = meta.get("timestamp")
    if not saved_at and not timestamp:
        t = now_kst()
        meta["saved_at"] = t
        meta["timestamp"] = t
        return
    if saved_at and not timestamp:
        meta["timestamp"] = saved_at
    elif timestamp and not saved_at:
        meta["saved_at"] = timestamp

def _ensure_model_signature(meta: dict):
    if not isinstance(meta.get("input_size"), int) or meta["input_size"] <= 0:
        meta["input_size"] = FEATURE_INPUT_SIZE
    if not isinstance(meta.get("num_classes"), int) or meta["num_classes"] <= 1:
        meta["num_classes"] = NUM_CLASSES

    cb = meta.get("class_bins", None)
    if cb is None:
        meta["class_bins"] = meta["num_classes"]
    else:
        if isinstance(cb, list):
            if len(cb) < 2:
                meta["class_bins"] = meta["num_classes"]
        elif isinstance(cb, int):
            if cb <= 1:
                meta["class_bins"] = meta["num_classes"]
        else:
            meta["class_bins"] = meta["num_classes"]

    try:
        mdl = get_model(
            meta.get("model", "lstm"),
            input_size=meta["input_size"],
            output_size=meta["num_classes"]
        )
        x = torch.randn(1, 20, meta["input_size"])
        if hasattr(mdl, "eval"): mdl.eval()
        with torch.no_grad():
            _ = mdl(x)
    except Exception as e:
        print(f"[⚠️ 모델 확인 실패] {meta.get('symbol','?')}-{meta.get('strategy','?')} "
              f"{meta.get('model','?')} → {e}")

def _find_existing_weight_by_stem(stem: str) -> str | None:
    """
    stem: /persistent/models/BTCUSDT_단기_lstm_group1_cls3 (확장자 없음)
    존재하는 가중치 파일을 선호 순서대로 검색 후 경로 반환.
    """
    for ext in _KNOWN_EXTS:
        cand = f"{stem}{ext}"
        if os.path.exists(cand):
            return cand
    # 디렉터리 별칭(SYMBOL/STRATEGY/{model}.{ext})도 탐색
    try:
        parts = os.path.basename(stem).split("_")
        if len(parts) >= 3:
            sym, strat, mtype = parts[0], parts[1], parts[2]
            for ext in _KNOWN_EXTS:
                cand = os.path.join(MODEL_DIR, sym, strat, f"{mtype}{ext}")
                if os.path.exists(cand):
                    return cand
    except Exception:
        pass
    # group/cls 와일드카드 버전도 마지막으로 탐색
    for ext in _KNOWN_EXTS:
        pat = f"{stem.split('_group')[0]}*{ext}"
        matches = sorted(glob.glob(os.path.join(MODEL_DIR, os.path.basename(pat))))
        for m in matches:
            if os.path.exists(os.path.join(MODEL_DIR, os.path.basename(m))):
                return os.path.join(MODEL_DIR, os.path.basename(m))
    return None

def _ensure_weight_fields(meta: dict, meta_filename: str):
    """
    메타에 들어있는 모델 파일 참조(예: model_name, weight_path 등)를
    실제 존재하는 확장자(.ptz/.safetensors/.pt)에 맞게 동기화.
    """
    stem = _stem_from_meta_filename(meta_filename)
    stem_abs = os.path.join(MODEL_DIR, stem)

    found = _find_existing_weight_by_stem(stem_abs)
    if not found:
        # 동일 stem이 아닐 수 있으니, meta 내부 힌트로 재시도
        hint = meta.get("model_name") or meta.get("weight_path") or meta.get("model_path")
        if isinstance(hint, str) and hint:
            hint_stem = os.path.join(MODEL_DIR, os.path.splitext(hint)[0])
            found = _find_existing_weight_by_stem(hint_stem)

    if not found:
        # 못 찾으면 조용히 패스(로깅만)
        print(f"[ℹ️ 가중치 미발견] {meta_filename} → stem='{stem}'")
        return

    bn = os.path.basename(found)
    # 표준 키들 동기화(있을 때만 업데이트)
    meta["model_name"] = bn
    meta.setdefault("weight_path", bn)   # 상대 경로로 유지
    meta.setdefault("model_path", bn)    # 호환 키

def _fill_defaults(meta: dict, fname_info: dict, meta_filename: str):
    # 파일명에서 파싱한 값 우선 반영
    for k in ["symbol", "strategy", "model", "group_id", "num_classes"]:
        if k not in meta or meta.get(k) in [None, ""]:
            if fname_info.get(k) is not None:
                meta[k] = fname_info[k]

    if "symbol" not in meta or not meta["symbol"]:
        meta["symbol"] = "UNKNOWN"

    if "strategy" not in meta or not meta["strategy"] or str(meta["strategy"]) not in _ALLOWED_STRATEGIES:
        meta["strategy"] = "단기"
    if "model" not in meta or not meta["model"] or str(meta["model"]) not in _ALLOWED_MODELS:
        meta["model"] = "lstm"

    if "group_id" not in meta or not isinstance(meta["group_id"], int):
        meta["group_id"] = 0

    if "saved_at" not in meta or not meta["saved_at"]:
        meta["saved_at"] = now_kst()

    _ensure_saved_ts_fields(meta)
    _ensure_model_signature(meta)
    _ensure_metrics(meta)
    _ensure_calibration(meta)
    _ensure_regime_cfg(meta)

    # ✅ NEW: 가중치 파일 참조 확장자 동기화
    _ensure_weight_fields(meta, meta_filename)

def _validate_and_fix(path: str, fname: str):
    """
    - 깨진/빈 json 복구
    - 필수 필드 보정
    - weight 확장자(.pt→.ptz/.safetensors 등) 동기화
    - 실패 시 안전 기본값으로 재생성(경고만 출력)
    """
    meta, err = _safe_load_json(path)
    fname_info = _parse_from_filename(fname)

    if meta is None:
        print(f"[⚠️ 손상 감지] {fname} → {err}. 안전 기본값으로 재생성")
        meta = {}
        _fill_defaults(meta, fname_info, fname)
        _safe_write_json(path, meta)
        return True, "created_defaults"

    changed = False
    if not isinstance(meta, dict):
        meta = {}
        changed = True

    pre = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    _fill_defaults(meta, fname_info, fname)
    post = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if pre != post:
        changed = True

    if changed:
        ok = _safe_write_json(path, meta)
        if ok:
            print(f"[FIXED] {fname} → 스키마/필드 보정 완료(확장자 동기화 포함)")
        return ok, "fixed"
    else:
        print(f"[OK] {fname} → 수정 불필요")
        return False, "ok"

def fix_all_meta_json():
    """
    /persistent/models 내 *.meta.json을 일괄 점검 및 복구.
    복구 실패해도 플로우는 계속(경고만 출력).
    """
    if not os.path.isdir(MODEL_DIR):
        print(f"[⚠️ 디렉터리 없음] {MODEL_DIR}")
        return

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".meta.json")]
    if not files:
        print("[ℹ️ 대상 meta.json 없음]")
        return

    for file in sorted(files):
        path = os.path.join(MODEL_DIR, file)
        try:
            _validate_and_fix(path, file)
        except Exception as e:
            print(f"[⚠️ 예외 발생] {file} → {e}. 안전 기본값 재생성 시도")
            meta = {}
            _fill_defaults(meta, _parse_from_filename(file), file)
            _safe_write_json(path, meta)

def check_meta_input_size():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".meta.json")]
    for file in sorted(files):
        path = os.path.join(MODEL_DIR, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            val = meta.get("input_size", None)
            if isinstance(val, int):
                print(f"[✅ 확인됨] {file} → input_size = {val}")
            else:
                print(f"[❌ 누락/비정상] {file} → input_size = {val}")
        except Exception as e:
            print(f"[ERROR] {file} 읽기 실패: {e}")

if __name__ == "__main__":
    fix_all_meta_json()
