# model_io.py (patched: file-lock, meta integrity, load/save robustness + strict=False default & rich errors)
from __future__ import annotations

import io
import os
import gzip
import json
import tempfile
import contextlib
import hashlib
import time
from typing import Any, Optional, Dict

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


def _to_state_dict(obj: Any) -> Dict[str, torch.Tensor] | Any:
    """
    ENFORCE_STATE_DICT_ONLY가 True면 가능한 경우 state_dict로 변환.
    - dict[str, Tensor] 그대로 통과
    - nn.Module 이면 .state_dict() 추출
    - OrderedDict-like 또는 mapping이면 텐서 값만 필터하여 dict로 반환
    - else 에러
    """
    if not ENFORCE_STATE_DICT_ONLY:
        return obj

    if isinstance(obj, dict):
        def _is_tensor_like(v):
            return hasattr(v, "shape") or hasattr(v, "size") or isinstance(v, torch.Tensor)
        if all(_is_tensor_like(v) for v in obj.values()):
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
    use_safetensors=True 를 주면 확장자가 .safetensors 가 아니면 ValueError.
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
            with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0) as gz:
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


def _load_raw(path: str, map_location: str | torch.device | None = "cpu") -> Any:
    """확장자에 맞게 '원본 저장물'을 복원(.pt/.ptz: torch 객체, .safetensors: 텐서 dict)"""
    if not os.path.isfile(path):
        raise ModelLoadError("file_not_found", path=path)

    ext = _ext(path)
    try:
        if ext == ".pt":
            with open(path, "rb") as f:
                return torch.load(f, map_location=map_location)

        if ext == ".ptz":
            with gzip.open(path, "rb") as gz:
                data = gz.read()
            return torch.load(io.BytesIO(data), map_location=map_location)

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


def _is_state_dict(obj: Any) -> bool:
    if isinstance(obj, dict):
        if not obj:
            return True
        def _is_tensor_like(v):
            return isinstance(v, torch.Tensor) or hasattr(v, "shape") or hasattr(v, "size")
        return all(_is_tensor_like(v) for v in obj.values())
    return obj.__class__.__name__.lower().endswith("ordereddict")


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    """module. 접두어 제거"""
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def load_model(
    path: str,
    model: Optional[nn.Module] = None,
    *,
    map_location: str | torch.device | None = "cpu",
    strict: bool = False,  # ✅ 기본 strict=False (키 불일치 허용)
) -> nn.Module | Dict[str, torch.Tensor] | Any:
    """
    통합 로더:
      - 파일 내용이 '모듈 전체'면 그 모듈을 그대로 반환
      - 파일 내용이 'state_dict(또는 safetensors 텐서 dict)'이면, 전달받은 `model`에 주입 후 `nn.Module` 반환
      - `model`이 None인데 state_dict인 경우: state_dict(또는 텐서 dict) 자체를 반환
    실패 시 ModelLoadError(reason=...)로 상세 사유 전달 → 상위(predict)에서 해당 모델만 스킵 가능.
    """
    try:
        raw = _load_raw(path, map_location=map_location)
    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError("load_io_error", path=path, detail=str(e))

    # 1) 저장물이 완성된 nn.Module인 경우
    if isinstance(raw, nn.Module):
        return raw

    # 2) 저장물이 state_dict(또는 safetensors 텐서 dict)인 경우
    if _is_state_dict(raw):
        if model is None:
            return raw
        # 시도 1: 전달된 strict로 로드
        try:
            model.load_state_dict(raw, strict=strict)
            return model
        except Exception as e1:
            # 시도 2: module. 접두어 제거
            try:
                fixed = _strip_module_prefix(raw)
                model.load_state_dict(fixed, strict=strict)
                return model
            except Exception as e2:
                # 시도 3: module. 접두어 추가
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

    # 3) 그 외 포맷(희귀) — 그대로 반환
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
    """
    모델 경로 옆에 `<stem>.meta.json`으로 원자적으로 저장.
    메타에 기본 필드(version, created_at, file_sha1) 추가.
    """
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
    """
    모델 경로 옆의 메타 파일을 읽어 dict로 반환. 없으면 default 또는 {} 반환.
    """
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
    """
    모델 저장 + (선택) 메타 저장을 한 번에 수행.
    """
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
