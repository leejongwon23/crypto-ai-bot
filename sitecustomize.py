# === sitecustomize.py (YOPO v1.0 — 경로자동변환 완성판) ===
# Render / Local 환경 어디서든 /persistent 경로를 안전하게 /opt/render/project/src/persistent 로 자동 변경
# 이제 모든 코드에서 경로를 일일이 수정할 필요 없음 👍

import os, builtins, os.path, shutil

# ✅ Render 환경변수에서 설정된 저장 경로 우선 사용
BASE = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/tmp/persistent"   # 없을 때 대비 기본값
)

def _fix(path):
    """모든 경로에서 /persistent → 환경변수 기반 BASE로 자동 변환"""
    import pathlib
    if isinstance(path, pathlib.Path):
        path = str(path)
    if isinstance(path, str) and path.startswith("/persistent"):
        return path.replace("/persistent", BASE, 1)
    return path

# 원본 함수 백업
_orig_open      = builtins.open
_orig_exists    = os.path.exists
_orig_isdir     = os.path.isdir
_orig_makedirs  = os.makedirs
_orig_listdir   = os.listdir
_orig_move      = shutil.move
_orig_remove    = os.remove
_orig_rmdir     = os.rmdir
_orig_rmtree    = shutil.rmtree

# === 패치 함수들 ===
def open_patched(path, *a, **kw):
    return _orig_open(_fix(path), *a, **kw)

def exists_patched(path):
    return _orig_exists(_fix(path))

def isdir_patched(path):
    return _orig_isdir(_fix(path))

def makedirs_patched(path, *a, **kw):
    return _orig_makedirs(_fix(path), *a, **kw)

def listdir_patched(path):
    return _orig_listdir(_fix(path))

def move_patched(src, dst, *a, **kw):
    return _orig_move(_fix(src), _fix(dst), *a, **kw)

def remove_patched(path, *a, **kw):
    return _orig_remove(_fix(path), *a, **kw)

def rmdir_patched(path, *a, **kw):
    return _orig_rmdir(_fix(path), *a, **kw)

def rmtree_patched(path, *a, **kw):
    return _orig_rmtree(_fix(path), *a, **kw)

# === 전역 덮어쓰기 ===
builtins.open   = open_patched
os.path.exists  = exists_patched
os.path.isdir   = isdir_patched
os.makedirs     = makedirs_patched
os.listdir      = listdir_patched
shutil.move     = move_patched
os.remove       = remove_patched
os.rmdir        = rmdir_patched
shutil.rmtree   = rmtree_patched

print(f"[sitecustomize] 경로자동변환 활성화됨 → BASE={BASE}")
