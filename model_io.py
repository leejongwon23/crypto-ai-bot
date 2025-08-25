# model_io.py — lossless compressed save/load wrapper for Torch models (pt / ptz / safetensors*)
# - ENFORCE_STATE_DICT_ONLY=True 면 어떤 입력이 와도 state_dict 로 강제 저장(용량 폭주 방지)
# - .ptz 는 gzip 무손실 압축
# - .safetensors 는 설치 시에만 사용(텐서 dict만)
from __future__ import annotations

import io
import os
import gzip
import tempfile
import contextlib
from typing import Any, Optional, Dict

import torch
import torch.nn as nn

# ===== 저장 정책 플래그 =====
ENFORCE_STATE_DICT_ONLY = True  # ← 프로젝트 전역에서 "state_dict만 저장"을 강제

# safetensors가 있으면 사용(없어도 동작)
try:
    from safetensors.torch import save_file as _st_save, load_file as _st_load  # type: ignore
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False

SUPPORTED_EXTS = {".pt", ".ptz", ".safetensors"}


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _atomic_write(bytes_data: bytes, dst_path: str) -> None:
    _ensure_dir(dst_path)
    dirpath = os.path.dirname(os.path.abspath(dst_path))
    with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tf:
        tmp = tf.name
        tf.write(bytes_data)
        tf.flush()
        os.fsync(tf.fileno())
    os.replace(tmp, dst_path)


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _to_state_dict(obj: Any) -> Dict[str, torch.Tensor] | Any:
    """
    ENFORCE_STATE_DICT_ONLY가 True면 어떤 입력이 와도 state_dict로 변환.
    - dict[str, Tensor] 그대로 통과
    - nn.Module 이면 .state_dict() 추출
    - 그 외는 에러(무분별한 객체 저장 방지)
    """
    if not ENFORCE_STATE_DICT_ONLY:
        return obj

    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict()
    raise ValueError("state_dict만 저장하도록 강제되었습니다. nn.Module을 전달했다면 .state_dict() 결과를 사용하세요.")


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

    # .safetensors 강제 요구 시 확장자 확인
    if use_safetensors is True and ext != ".safetensors":
        raise ValueError("use_safetensors=True 일 때는 경로 확장자가 .safetensors 여야 합니다.")

    # ===== 입력을 정책에 맞게 정규화 (state_dict 강제) =====
    obj = _to_state_dict(state_or_obj)

    # ===== .pt : 원형 torch.save =====
    if ext == ".pt":
        buf = io.BytesIO()
        torch.save(obj, buf)
        _atomic_write(buf.getvalue(), path)
        return

    # ===== .ptz : torch.save 결과를 gzip으로 무손실 압축 =====
    if ext == ".ptz":
        raw = io.BytesIO()
        torch.save(obj, raw)
        raw_bytes = raw.getvalue()
        gz_buf = io.BytesIO()
        # mtime=0 → gzip 헤더 고정(바이트 결정적) / 무손실
        with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0) as gz:
            gz.write(raw_bytes)
        _atomic_write(gz_buf.getvalue(), path)
        return

    # ===== .safetensors : 설치되어 있을 때만 사용(텐서 dict만) =====
    if ext == ".safetensors":
        if not _HAVE_ST:
            raise RuntimeError("safetensors 미설치: `pip install safetensors` 후 사용하거나 .ptz를 사용하세요.")
        # safetensors는 텐서 dict만 지원
        tensors: Dict[str, torch.Tensor]
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            tensors = obj  # 이미 텐서 dict
        else:
            # state_dict에 텐서 외 값이 섞여 있으면 방지
            if hasattr(obj, "items"):
                if not all(isinstance(v, torch.Tensor) for _, v in obj.items()):
                    raise ValueError("safetensors 저장은 텐서 dict만 지원합니다.")
                tensors = dict(obj)
            else:
                raise ValueError("safetensors 저장은 텐서 dict만 지원합니다.")
        _ensure_dir(path)
        dirpath = os.path.dirname(os.path.abspath(path))
        with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tf:
            tmp = tf.name
        try:
            _st_save(tensors, tmp)
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(Exception):
                os.remove(tmp)
            raise
        return


def _load_raw(path: str, map_location: str | torch.device | None = "cpu") -> Any:
    """확장자에 맞게 '원본 저장물'을 복원(.pt/.ptz: torch 객체, .safetensors: 텐서 dict)"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = _ext(path)
    if ext == ".pt":
        with open(path, "rb") as f:
            return torch.load(f, map_location=map_location)

    if ext == ".ptz":
        with gzip.open(path, "rb") as gz:
            data = gz.read()
        return torch.load(io.BytesIO(data), map_location=map_location)

    if ext == ".safetensors":
        if not _HAVE_ST:
            raise RuntimeError("safetensors 미설치: .safetensors 파일을 읽으려면 `pip install safetensors`가 필요합니다.")
        # safetensors는 텐서 dict만 반환
        device = map_location if isinstance(map_location, str) else "cpu"
        return _st_load(path, device=device)

    raise ValueError(f"Unsupported extension: {ext}")


def _is_state_dict(obj: Any) -> bool:
    # torch.save로 저장된 state_dict(dict[str, Tensor]) 또는 OrderedDict
    if isinstance(obj, dict):
        return all(isinstance(v, torch.Tensor) for v in obj.values()) or \
               all(hasattr(v, "size") for v in obj.values())
    # pytorch의 OrderedDict 타입명으로 오는 경우 방어
    return obj.__class__.__name__.lower().endswith("ordereddict")


def load_model(
    path: str,
    model: Optional[nn.Module] = None,
    *,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> nn.Module | Dict[str, torch.Tensor] | Any:
    """
    통합 로더:
      - 파일 내용이 '모듈 전체'면 그 모듈을 그대로 반환
      - 파일 내용이 'state_dict(또는 safetensors 텐서 dict)'이면, 전달받은 `model`에 주입 후 `nn.Module` 반환
      - `model`이 None인데 state_dict인 경우: state_dict(또는 텐서 dict) 자체를 반환
    """
    raw = _load_raw(path, map_location=map_location)

    # 1) 저장물이 완성된 nn.Module인 경우(전체 직렬화). (과거 파일 호환)
    if isinstance(raw, nn.Module):
        return raw

    # 2) 저장물이 state_dict(또는 safetensors 텐서 dict)인 경우
    if _is_state_dict(raw):
        if model is None:
            return raw  # 호출부에서 직접 처리 가능
        try:
            model.load_state_dict(raw, strict=strict)
        except RuntimeError:
            # DataParallel 'module.' prefix mismatch 등 호환 보정
            if any(k.startswith("module.") for k in raw.keys()):
                fixed = {k.replace("module.", "", 1): v for k, v in raw.items()}
            else:
                fixed = {("module." + k): v for k, v in raw.items()}
            model.load_state_dict(fixed, strict=strict)
        return model

    # 3) 그 외 포맷(희귀) — 그대로 반환(호출자가 처리)
    return raw


def convert_pt_to_ptz(
    src_path: str,
    dst_path: Optional[str] = None,
    *,
    map_location: str | torch.device | None = "cpu",
) -> str:
    """
    기존 .pt 파일을 .ptz로 무손실 변환(내용 동일, 용량만 축소).
    반환: 생성된 .ptz 경로
    """
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
    """
    .ptz를 풀어서 .pt로 변환(무손실).
    """
    if _ext(src_path) != ".ptz":
        raise ValueError("src_path는 .ptz 여야 합니다.")
    obj = _load_raw(src_path, map_location=map_location)
    buf = io.BytesIO()
    torch.save(obj, buf)
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".pt"
    _atomic_write(buf.getvalue(), dst_path)
    return dst_path
