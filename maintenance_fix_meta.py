# maintenance_fix_meta.py (FINAL, 12번 반영: calibration/regime_cfg 점검·복구)
import os
import json
import re
import datetime
import pytz
import torch

from model.base_model import get_model  # 현재 모델 구조 확인용
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE

NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

MODEL_DIR = "/persistent/models"

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

def _parse_from_filename(fname: str):
    base = fname.replace(".meta.json", "")
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
    # 최소 스키마
    if "val_acc" not in metrics or not isinstance(metrics.get("val_acc"), (int, float)):
        metrics["val_acc"] = 0.0
    # 있으면 그대로 두고, 없으면 기본 추가(호환)
    if "val_f1" not in metrics or not isinstance(metrics.get("val_f1"), (int, float)):
        metrics["val_f1"] = 0.0
    if "train_loss_sum" not in metrics or not isinstance(metrics.get("train_loss_sum"), (int, float)):
        metrics["train_loss_sum"] = 0.0
    meta["metrics"] = metrics

def _ensure_calibration(meta: dict):
    """
    calibration 블록이 없거나 파손되면 안전 기본값으로 복구.
    필드:
      method: "none" | "temperature" | "platt"
      temperature: float
      platt: {"a": float, "b": float}
      updated_at: ISO string
      ver: int
    """
    cal = meta.get("calibration")
    if not isinstance(cal, dict):
        cal = {}
    method = cal.get("method", "none")
    if method not in ["none", "temperature", "platt"]:
        method = "none"
    # temperature
    try:
        temperature = float(cal.get("temperature", 1.0))
        if temperature <= 0 or not (temperature == temperature):  # nan 체크
            temperature = 1.0
    except Exception:
        temperature = 1.0
    # platt 계수
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
    # 버전/업데이트 시간
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
    """
    regime_cfg 블록이 없거나 파손되면 안전 기본값으로 복구.
    필드(경량):
      enabled: bool
      detector: "none" | "simple" | "volatility" (등, 프로젝트 내부 규약만 준수)
      params: dict
      ver: int
    """
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

def _ensure_model_signature(meta: dict):
    """
    모델 구조를 불러 input_size/num_classes 정합성만 가볍게 확인.
    실패해도 플로우 끊지 않음.
    """
    # input_size
    if not isinstance(meta.get("input_size"), int) or meta["input_size"] <= 0:
        meta["input_size"] = FEATURE_INPUT_SIZE

    # num_classes
    if not isinstance(meta.get("num_classes"), int) or meta["num_classes"] <= 1:
        meta["num_classes"] = NUM_CLASSES

    # class_bins: 프로젝트마다 의미가 다를 수 있어 안전 기본값 사용
    # int 또는 리스트 모두 허용. 없으면 num_classes로 대체.
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

    # 모델 로드 시도(선택). 실패해도 경고만.
    try:
        mdl = get_model(
            meta.get("model", "lstm"),
            input_size=meta["input_size"],
            output_size=meta["num_classes"]
        )
        # 아주 가벼운 더미 포워드(창 길이는 임의 20)
        x = torch.randn(1, 20, meta["input_size"])
        if hasattr(mdl, "eval"):
            mdl.eval()
        with torch.no_grad():
            _ = mdl(x)
    except Exception as e:
        print(f"[⚠️ 모델 확인 실패] {meta.get('symbol','?')}-{meta.get('strategy','?')} "
              f"{meta.get('model','?')} → {e}")

def _fill_defaults(meta: dict, fname_info: dict):
    # 파일명에서 파싱한 값 우선 반영
    for k in ["symbol", "strategy", "model", "group_id", "num_classes"]:
        if k not in meta or meta.get(k) in [None, ""]:
            if fname_info.get(k) is not None:
                meta[k] = fname_info[k]

    # 기본값
    if "symbol" not in meta or not meta["symbol"]:
        meta["symbol"] = "UNKNOWN"
    if "strategy" not in meta or not meta["strategy"]:
        meta["strategy"] = "단기"  # 안전 기본값
    if "model" not in meta or not meta["model"]:
        meta["model"] = "lstm"
    if "group_id" not in meta or not isinstance(meta["group_id"], int):
        meta["group_id"] = 0
    if "saved_at" not in meta or not meta["saved_at"]:
        meta["saved_at"] = now_kst()

    _ensure_model_signature(meta)
    _ensure_metrics(meta)
    _ensure_calibration(meta)  # ✅ 12번 요구사항
    _ensure_regime_cfg(meta)   # ✅ 12번 요구사항

def _validate_and_fix(path: str, fname: str):
    """
    - 깨진/빈 json 복구
    - 필수 필드 보정
    - 실패 시 안전 기본값으로 재생성(경고만 출력)
    """
    meta, err = _safe_load_json(path)
    fname_info = _parse_from_filename(fname)

    if meta is None:
        print(f"[⚠️ 손상 감지] {fname} → {err}. 안전 기본값으로 재생성")
        meta = {}
        _fill_defaults(meta, fname_info)
        _safe_write_json(path, meta)
        return True, "created_defaults"

    # 정상 로드된 경우에도 필수 필드 보정
    changed = False

    # 빈 오브젝트/부분 손상 방지
    if not isinstance(meta, dict):
        meta = {}
        changed = True

    # 누락/잘못된 값 보정
    pre = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    _fill_defaults(meta, fname_info)
    post = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    if pre != post:
        changed = True

    if changed:
        ok = _safe_write_json(path, meta)
        if ok:
            print(f"[FIXED] {fname} → 스키마/필드 보정 완료")
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
            # 최후 안전장치: 예외가 나도 기본값으로 덮어써서 진행
            print(f"[⚠️ 예외 발생] {file} → {e}. 안전 기본값 재생성 시도")
            meta = {}
            _fill_defaults(meta, _parse_from_filename(file))
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
