# model_io.py (final, robust load for prediction + finetune loader)

from __future__ import annotations

import io
import os
import gzip
import json
import tempfile
import contextlib
import hashlib
import time
from typing import Any, Optional, Dict, Tuple, List

import torch
import torch.nn as nn

# ===== 저장 정책 플래그 =====
ENFORCE_STATE_DICT_ONLY = True  # 프로젝트 전역에서 "state_dict만 저장"을 강제

# ===== 명확한 실패 사유 전달을 위한 예외 =====
class ModelLoadError(RuntimeError):
    def __init__(self, reason: str, *, path: str = "", detail: str = ""):
        super().__init__(f"{reason} :: {path} :: {detail}")
        self.reason = reason
        self.path = path
        self.detail = detail

# safetensors가 있으면 사용(없어도 동작)
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load  # type: ignore
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False

SUPPORTED_EXTS = {".pt", ".ptz", ".safetensors"}

# 환경: gzip 압축 레벨 (0-9), 기본 6
_GZIP_LEVEL = int(os.getenv("MODEL_IO_GZIP_LEVEL", "6"))
# 메타 버전
_META_VERSION = 1


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _atomic_write(bytes_data: bytes, dst_path: str) -> None:
    _ensure_dir(dst_path)
    dirpath = os.path.dirname(os.path.abspath(dst_path))
    with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tf:
        tmp = tf.name
        tf.write(bytes_data)
        tf.flush()
        with contextlib.suppress(Exception):
            os.fsync(tf.fileno())
    os.replace(tmp, dst_path)


def _atomic_write_json(dst_path: str, obj: dict) -> None:
    _ensure_dir(dst_path)
    dirpath = os.path.dirname(os.path.abspath(dst_path))
    with tempfile.NamedTemporaryFile(dir=dirpath, suffix=".tmp", delete=False, mode="w", encoding="utf-8") as tf:
        tmp = tf.name
        json.dump(obj, tf, ensure_ascii=False, indent=2)
        tf.flush()
        with contextlib.suppress(Exception):
            os.fsync(tf.fileno())
    os.replace(tmp, dst_path)


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _stem(path: str) -> str:
    base, _ = os.path.splitext(os.path.abspath(path))
    return base


def _meta_path_for(model_path: str) -> str:
    return _stem(model_path) + ".meta.json"


# 간단 파일락 (멀티프로세스 원자성 보조)
_LOCK_STALE_SEC = 120
class _FileLock:
    def __init__(self, path: str, timeout: float = 10.0, poll: float = 0.05):
        self.path = path + ".lock"
        self.timeout = float(timeout)
        self.poll = float(poll)

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                if os.path.exists(self.path):
                    with contextlib.suppress(Exception):
                        mtime = os.path.getmtime(self.path)
                        if (time.time() - mtime) > _LOCK_STALE_SEC:
                            os.remove(self.path)
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} ts={time.time()}\n")
                break
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"lock timeout: {self.path}")
                time.sleep(self.poll)
        return self

    def __exit__(self, exc_type, exc, tb):
        with contextlib.suppress(Exception):
            if os.path.exists(self.path):
                os.remove(self.path)


def _tensor_like(v: Any) -> bool:
    return isinstance(v, torch.Tensor) or hasattr(v, "shape") or hasattr(v, "size")


def _to_state_dict(obj: Any) -> Dict[str, torch.Tensor] | Any:
    """
    ENFORCE_STATE_DICT_ONLY가 True면 가능한 경우 state_dict로 변환.
    """
    if not ENFORCE_STATE_DICT_ONLY:
        return obj

    if isinstance(obj, dict):
        if all(_tensor_like(v) for v in obj.values()):
            return obj
        raise ValueError("ENFORCE_STATE_DICT_ONLY=True: dict contains non-tensor values; pass state_dict() with tensors only.")

    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict()

    name = getattr(obj, "__class__", None)
    if name and getattr(obj.__class__, "__name__", "").lower().endswith("ordereddict"):
        return dict(obj)

    raise ValueError("state_dict만 저장하도록 강제되었습니다. nn.Module을 전달하거나 텐서 dict를 전달하세요.")


def save_model(path: str, state_or_obj: Any, *, use_safetensors: Optional[bool] = None) -> None:
    """
    path 확장자로 포맷 결정:
      - .pt       : torch.save (무압축, state_dict 강제)
      - .ptz      : torch.save + gzip 무손실 압축 (권장)
      - .safetensors : safetensors (설치 시), 텐서 dict만 허용
    """
    ext = _ext(path)
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTS))}")

    if use_safetensors is True and ext != ".safetensors":
        raise ValueError("use_safetensors=True 일 때는 경로 확장자가 .safetensors 여야 합니다.")

    obj = _to_state_dict(state_or_obj)

    with _FileLock(path, timeout=10.0):
        if ext == ".pt":
            buf = io.BytesIO()
            torch.save(obj, buf)
            _atomic_write(buf.getvalue(), path)
            return

        if ext == ".ptz":
            raw = io.BytesIO()
            torch.save(obj, raw)
            raw_bytes = raw.getvalue()
            gz_buf = io.BytesIO()
            with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0, compresslevel=_GZIP_LEVEL) as gz:
                gz.write(raw_bytes)
            _atomic_write(gz_buf.getvalue(), path)
            return

        if ext == ".safetensors":
            if not _HAVE_ST:
                raise RuntimeError("safetensors 미설치: `pip install safetensors` 후 사용하거나 .ptz를 사용하세요.")
            if not isinstance(obj, dict) or not all(isinstance(v, torch.Tensor) for v in obj.values()):
                raise ValueError("safetensors 저장은 텐서 dict만 지원합니다.")
            _ensure_dir(path)
            dirpath = os.path.dirname(os.path.abspath(path))
            with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tf:
                tmp = tf.name
            try:
                _st_save(obj, tmp)
                os.replace(tmp, path)
            except Exception:
                with contextlib.suppress(Exception):
                    os.remove(tmp)
                raise


def _is_state_dict(obj: Any) -> bool:
    if isinstance(obj, dict):
        if not obj:
            return True
        return all(_tensor_like(v) for v in obj.values())
    return obj.__class__.__name__.lower().endswith("ordereddict")


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def _try_torch_load_bytes(data: bytes, map_location: str | torch.device | None = "cpu") -> Any | None:
    for loc in (map_location, "cpu"):
        try:
            return torch.load(io.BytesIO(data), map_location=loc)
        except Exception:
            continue
    return None


def _try_load_ptz_with_fallbacks(path: str, map_location: str | torch.device | None = "cpu") -> Any:
    try:
        with gzip.open(path, "rb") as gz:
            data = gz.read()
        obj = _try_torch_load_bytes(data, map_location)
        if obj is not None:
            return obj
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            raw = f.read()
        obj = _try_torch_load_bytes(raw, map_location)
        if obj is not None:
            return obj
    except Exception:
        pass

    raise ModelLoadError("raw_load_failed", path=path, detail="ptz decode failed (gzip/torch.load)")


def _load_raw(path: str, map_location: str | torch.device | None = "cpu") -> Any:
    if not os.path.isfile(path):
        raise ModelLoadError("file_not_found", path=path)

    ext = _ext(path)
    try:
        if ext == ".pt":
            with open(path, "rb") as f:
                return torch.load(f, map_location=map_location)

        if ext == ".ptz":
            return _try_load_ptz_with_fallbacks(path, map_location)

        if ext == ".safetensors":
            if not _HAVE_ST:
                raise ModelLoadError("safetensors_not_installed", path=path, detail="pip install safetensors")
            device = map_location if isinstance(map_location, (str, torch.device)) else "cpu"
            return _st_load(path, device=device)

    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError("raw_load_failed", path=path, detail=str(e))

    raise ModelLoadError("unsupported_extension", path=path, detail=ext)


def _coerce_to_state_dict(raw: Any) -> Dict[str, torch.Tensor] | None:
    if _is_state_dict(raw):
        return raw
    if isinstance(raw, nn.Module):
        try:
            return raw.state_dict()
        except Exception:
            return None
    if hasattr(raw, "state_dict") and callable(getattr(raw, "state_dict")):
        try:
            return raw.state_dict()
        except Exception:
            return None
    if raw.__class__.__name__.lower().endswith("ordereddict"):
        try:
            d = dict(raw)
            if all(_tensor_like(v) for v in d.values()):
                return d
        except Exception:
            pass
    return None


def load_model(
    path: str,
    model: Optional[nn.Module] = None,
    *,
    map_location: str | torch.device | None = "cpu",
    strict: bool = False,
) -> nn.Module | Dict[str, torch.Tensor] | Any:
    try:
        raw = _load_raw(path, map_location=map_location)
    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError("load_io_error", path=path, detail=str(e))

    if isinstance(raw, nn.Module):
        if model is None:
            return raw
        try:
            model.load_state_dict(raw.state_dict(), strict=strict)
            return model
        except Exception as e:
            try:
                fixed = _strip_module_prefix(raw.state_dict())
                model.load_state_dict(fixed, strict=strict)
                return model
            except Exception as e2:
                try:
                    augmented = {("module." + k): v for k, v in raw.state_dict().items()}
                    model.load_state_dict(augmented, strict=strict)
                    return model
                except Exception as e3:
                    raise ModelLoadError(
                        "state_dict_from_module_failed",
                        path=path,
                        detail=f"e1={type(e).__name__}, e2={type(e2).__name__}, e3={type(e3).__name__}",
                    )

    if _is_state_dict(raw):
        if model is None:
            return raw
        try:
            model.load_state_dict(raw, strict=strict)
            return model
        except Exception as e1:
            try:
                fixed = _strip_module_prefix(raw)
                model.load_state_dict(fixed, strict=strict)
                return model
            except Exception as e2:
                try:
                    augmented = {("module." + k): v for k, v in raw.items()}
                    model.load_state_dict(augmented, strict=strict)
                    return model
                except Exception as e3:
                    raise ModelLoadError(
                        "state_dict_load_failed",
                        path=path,
                        detail=f"e1={type(e1).__name__}, e2={type(e2).__name__}, e3={type(e3).__name__}"
                    )

    coerced = _coerce_to_state_dict(raw)
    if coerced is not None:
        if model is None:
            return coerced
        try:
            model.load_state_dict(coerced, strict=strict)
            return model
        except Exception as e1:
            try:
                fixed = _strip_module_prefix(coerced)
                model.load_state_dict(fixed, strict=strict)
                return model
            except Exception as e2:
                try:
                    augmented = {("module." + k): v for k, v in coerced.items()}
                    model.load_state_dict(augmented, strict=strict)
                    return model
                except Exception as e3:
                    raise ModelLoadError(
                        "state_dict_coerce_failed",
                        path=path,
                        detail=f"e1={type(e1).__name__}, e2={type(e2).__name__}, e3={type(e3).__name__}"
                    )

    return raw


# ========================= 메타 저장/로드 (사이드카) =========================
def _compute_sha1_of_file(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def save_meta(model_path: str, meta: Dict[str, Any]) -> str:
    if not isinstance(meta, dict):
        raise ValueError("meta는 dict여야 합니다.")
    mpath = _meta_path_for(model_path)
    base = dict(meta)
    base.setdefault("meta_version", _META_VERSION)
    base.setdefault("created_at", time.time())
    try:
        base["file_sha1"] = _compute_sha1_of_file(model_path) if os.path.exists(model_path) else ""
    except Exception:
        base["file_sha1"] = ""
    _atomic_write_json(mpath, base)
    return mpath


def load_meta(model_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mpath = _meta_path_for(model_path)
    if not os.path.isfile(mpath):
        return {} if default is None else dict(default)
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {} if default is None else dict(default)
        return obj
    except Exception:
        return {} if default is None else dict(default)


def save_model_with_meta(
    path: str,
    state_or_obj: Any,
    meta: Optional[Dict[str, Any]] = None,
    *,
    use_safetensors: Optional[bool] = None,
) -> None:
    save_model(path, state_or_obj, use_safetensors=use_safetensors)
    if meta:
        save_meta(path, meta)


def convert_pt_to_ptz(
    src_path: str,
    dst_path: Optional[str] = None,
    *,
    map_location: str | torch.device | None = "cpu",
) -> str:
    if _ext(src_path) != ".pt":
        raise ValueError("src_path는 .pt 여야 합니다.")
    obj = _load_raw(src_path, map_location=map_location)
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".ptz"
    save_model(dst_path, obj)
    return dst_path


def convert_ptz_to_pt(
    src_path: str,
    dst_path: Optional[str] = None,
    *,
    map_location: str | torch.device | None = "cpu",
) -> str:
    if _ext(src_path) != ".ptz":
        raise ValueError("src_path는 .ptz 여야 합니다.")
    obj = _load_raw(src_path, map_location=map_location)
    buf = io.BytesIO()
    torch.save(obj, buf)
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".pt"
    _atomic_write(buf.getvalue(), dst_path)
    return dst_path


# ========================= 파인튜닝 전용 로더 =========================
def _filter_backbone_keys(
    state: Dict[str, torch.Tensor],
    drop_head: bool,
    head_prefixes: Tuple[str, ...]
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    kept: Dict[str, torch.Tensor] = {}
    kept_keys: List[str] = []
    dropped: List[str] = []
    for k, v in state.items():
        top = k.split(".", 1)[0]
        if drop_head and any(top.startswith(h) or k.startswith(h) for h in head_prefixes):
            dropped.append(k)
            continue
        kept[k] = v
        kept_keys.append(k)
    return kept, kept_keys, dropped


def load_for_finetune(
    model: nn.Module,
    prev_path: str,
    *,
    strict: bool = False,
    map_location: str | torch.device | None = "cpu",
    drop_head: bool = True,
    head_prefixes: Tuple[str, ...] = ("fc_logits", "fc2", "fc1", "res_proj"),
) -> Dict[str, Any]:
    """
    이전 모델 가중치를 '백본 중심'으로 주입.
    - 헤드 계층(head_prefixes)은 기본적으로 제외(drop_head=True).
    - 키 접두어(module.) 처리 및 크기 불일치 자동 스킵.
    반환: {"loaded": [...], "skipped": [...], "dropped": [...], "path": prev_path}
    """
    # 1) raw 로드 -> state_dict 확보
    raw = load_model(prev_path, model=None, map_location=map_location, strict=False)
    state = _coerce_to_state_dict(raw)
    if state is None and _is_state_dict(raw):
        state = raw  # type: ignore
    if state is None:
        raise ModelLoadError("finetune_state_missing", path=prev_path, detail="no state_dict coercible")

    state = _strip_module_prefix(state)  # module. 제거

    # 2) 헤드 제외 필터링
    filt_state, kept_keys, dropped = _filter_backbone_keys(state, drop_head, head_prefixes)

    # 3) 크기 불일치 키 제거
    own_state = dict(model.state_dict())
    loadable: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in filt_state.items():
        if k not in own_state:
            skipped.append(k)
            continue
        try:
            if own_state[k].shape != v.shape:
                skipped.append(k)
                continue
        except Exception:
            skipped.append(k)
            continue
        loadable[k] = v

    # 4) 주입
    missing, unexpected = [], []
    try:
        res = model.load_state_dict(loadable, strict=strict)
        missing = list(getattr(res, "missing_keys", [])) if hasattr(res, "missing_keys") else []
        unexpected = list(getattr(res, "unexpected_keys", [])) if hasattr(res, "unexpected_keys") else []
    except Exception:
        # 안전 폴백: strict=False 재시도
        model.load_state_dict(loadable, strict=False)

    return {
        "path": prev_path,
        "loaded": sorted(list(loadable.keys())),
        "skipped": sorted(skipped + unexpected),
        "dropped": sorted(dropped),
        "missing": sorted(missing),
    }


# === NEW: 엄격 매칭 유틸리티 (선택/검증 전용, 기존 로더 무변경) =================
def is_checkpoint_compatible(
    meta: Dict[str, Any],
    *,
    symbol: str,
    horizon: str,   # "단기" | "중기" | "장기" 등 프로젝트 표기
    model: str,     # "lstm" | "cnn_lstm" | "transformer" 등
) -> bool:
    """
    사이드카 메타 기준으로 정확 일치 여부를 판단.
    필수 키: symbol, horizon, model  (대소문자/공백 보정 포함)
    """
    try:
        def norm(x: Any) -> str:
            return str(x).strip().lower()
        return (
            norm(meta.get("symbol"))  == norm(symbol)
            and norm(meta.get("horizon")) == norm(horizon)
            and norm(meta.get("model"))   == norm(model)
        )
    except Exception:
        return False


def _safe_get_created_at(meta: Dict[str, Any]) -> float:
    v = meta.get("created_at")
    try:
        return float(v)
    except Exception:
        return 0.0


def resolve_checkpoint_strict(
    search_dirs: List[str] | Tuple[str, ...],
    *,
    symbol: str,
    horizon: str,
    model: str,
    exts: set[str] = SUPPORTED_EXTS,
) -> Tuple[Optional[str], str]:
    """
    주어진 디렉터리 집합에서 사이드카 메타를 이용해
    (symbol, horizon, model) 완전 일치하는 ckpt만 선택.
    - 메타 누락/해시 불일치/키 불완전 → 후보 제외
    - 다수 후보면 created_at(메타) → 파일 mtime 순으로 최신 선택
    반환: (path or None, reason_code)
      reason_code ∈ {"ok", "not_found", "no_meta", "hash_mismatch"}
      (여러 사유가 섞일 수 있으나 최종 우선순위의 사유만 표기)
    """
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    seen_no_meta = False
    seen_hash_miss = False

    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if not os.path.isfile(p) or _ext(p) not in exts:
                continue
            mpath = _meta_path_for(p)
            if not os.path.isfile(mpath):
                seen_no_meta = True
                continue
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if not isinstance(meta, dict):
                    continue
            except Exception:
                continue

            if not is_checkpoint_compatible(meta, symbol=symbol, horizon=horizon, model=model):
                continue

            # 파일 해시 일치성 확인(무결성 보장)
            try:
                file_sha1 = meta.get("file_sha1") or ""
                if file_sha1:
                    cur_sha1 = _compute_sha1_of_file(p)
                    if not cur_sha1 or cur_sha1 != file_sha1:
                        seen_hash_miss = True
                        continue
            except Exception:
                seen_hash_miss = True
                continue

            candidates.append((p, meta))

    if not candidates:
        if seen_hash_miss:
            return None, "hash_mismatch"
        if seen_no_meta:
            return None, "no_meta"
        return None, "not_found"

    # 최신 우선: created_at(desc) → 파일 mtime(desc)
    candidates.sort(key=lambda pm: (_safe_get_created_at(pm[1]), os.path.getmtime(pm[0])), reverse=True)
    return candidates[0][0], "ok"
# === NEW END ================================================================


__all__ = [
    "ModelLoadError",
    "save_model",
    "load_model",
    "save_meta",
    "load_meta",
    "save_model_with_meta",
    "convert_pt_to_ptz",
    "convert_ptz_to_pt",
    "load_for_finetune",
    # NEW helpers
    "is_checkpoint_compatible",
    "resolve_checkpoint_strict",
            ]
