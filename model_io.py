# model_io.py — lossless compressed save/load wrapper for Torch models (pt / ptz / safetensors*)
# *safetensors 사용은 설치되어 있을 때만 활성화됩니다. (기능 동일, 무손실)

from __future__ import annotations
import io, os, gzip, tempfile, shutil
from typing import Any, Optional, Dict

import torch

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

def save_model(path: str, state_or_obj: Any, *, use_safetensors: Optional[bool] = None) -> None:
    """
    path 확장자로 포맷 결정:
      - .pt       : torch.save (무압축)
      - .ptz      : torch.save + gzip 무손실 압축
      - .safetensors : safetensors (설치 시), 텐서 dict만 허용
    use_safetensors=True를 주면 확장자가 .safetensors가 아니면 ValueError
    """
    ext = _ext(path)
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTS))}")

    # .safetensors 강제 요구 시 확장자 확인
    if use_safetensors is True and ext != ".safetensors":
        raise ValueError("use_safetensors=True 일 때는 경로 확장자가 .safetensors 여야 합니다.")

    # ===== .pt : 원형 torch.save =====
    if ext == ".pt":
        # 그대로 직렬화(무손실/무압축)
        buf = io.BytesIO()
        torch.save(state_or_obj, buf)
        _atomic_write(buf.getvalue(), path)
        return

    # ===== .ptz : torch.save 결과를 gzip으로 무손실 압축 =====
    if ext == ".ptz":
        raw = io.BytesIO()
        torch.save(state_or_obj, raw)
        raw_bytes = raw.getvalue()
        # gzip mtime=0으로 재현성 향상
        gz_buf = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0) as gz:
            gz.write(raw_bytes)
        _atomic_write(gz_buf.getvalue(), path)
        return

    # ===== .safetensors : 설치되어 있을 때만 사용 =====
    if ext == ".safetensors":
        if not _HAVE_ST:
            raise RuntimeError("safetensors 미설치: `pip install safetensors` 후 사용하거나 .ptz를 사용하세요.")
        # safetensors는 텐서 dict만 지원 → state_dict 형태 권장
        tensors: Dict[str, torch.Tensor]
        if isinstance(state_or_obj, dict) and all(isinstance(v, torch.Tensor) for v in state_or_obj.values()):
            tensors = state_or_obj  # 이미 텐서 dict
        elif hasattr(state_or_obj, "state_dict"):
            sd = state_or_obj.state_dict()
            if not all(isinstance(v, torch.Tensor) for v in sd.values()):
                raise ValueError("safetensors 저장은 텐서 dict만 지원합니다. state_dict에 텐서 외 값이 포함됨.")
            tensors = sd
        else:
            raise ValueError("safetensors 저장은 텐서 dict 또는 state_dict를 지원합니다.")
        # safetensors는 파일에 직접 씀 → 원자적 쓰기를 위해 임시 파일 사용
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

def load_model(path: str, *, map_location: str | torch.device | None = "cpu") -> Any:
    """
    경로 확장자를 자동 인식해서 무손실 복원:
      - .pt       : torch.load
      - .ptz      : gzip 해제 후 torch.load
      - .safetensors : safetensors.load_file → 텐서 dict 반환
    반환값:
      - .pt/.ptz : torch.save에 넣었던 원본 객체(모듈/ state_dict 등)
      - .safetensors : 텐서 dict
    """
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
        return _st_load(path, device=map_location if isinstance(map_location, str) else "cpu")

    raise ValueError(f"Unsupported extension: {ext}")

def convert_pt_to_ptz(src_path: str, dst_path: Optional[str] = None, *, map_location: str | torch.device | None = "cpu") -> str:
    """
    기존 .pt 파일을 .ptz로 무손실 변환(내용 동일, 용량만 축소).
    반환: 생성된 .ptz 경로
    """
    if _ext(src_path) != ".pt":
        raise ValueError("src_path는 .pt 여야 합니다.")
    obj = load_model(src_path, map_location=map_location)
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".ptz"
    save_model(dst_path, obj)
    return dst_path

def convert_ptz_to_pt(src_path: str, dst_path: Optional[str] = None, *, map_location: str | torch.device | None = "cpu") -> str:
    """
    .ptz를 풀어서 .pt로 변환(무손실).
    """
    if _ext(src_path) != ".ptz":
        raise ValueError("src_path는 .ptz 여야 합니다.")
    obj = load_model(src_path, map_location=map_location)
    if dst_path is None:
        dst_path = os.path.splitext(src_path)[0] + ".pt"
    # obj가 state_dict든 모듈이든 torch.save로 동일하게 복원됨
    buf = io.BytesIO()
    torch.save(obj, buf)
    _atomic_write(buf.getvalue(), dst_path)
    return dst_path
