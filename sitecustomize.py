# === sitecustomize.py (YOPO v1.1 — 경로자동변환 + 기본폴더 보장) ===
# 목적:
# 1) 코드 어디에서든 /persistent/... 라고 쓰면 실제로는 BASE/... 으로 가게 한다
# 2) 그리고 그 BASE랑 우리가 자주 쓰는 하위폴더를 미리 만들어둔다
#    → "No such file or directory: '/persistent/...csv'" 이런 거 안 나게

import os, builtins, os.path, shutil

# ✅ Render / 로컬에서 덮어쓸 수 있는 기본 루트
BASE = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/tmp/persistent"   # 아무것도 없을 때 마지막 기본값
)

# ✅ 여기서 한 번 실제 디렉터리를 만들어둔다
#    (파일 열 때마다 매번 실패하는 것보다 한번 만드는 게 낫다)
try:
    os.makedirs(BASE, exist_ok=True)
    # 우리가 자주 쓰는 서브디렉터리도 같이 만들어둠
    for sub in ("logs", "state", "locks", "importances", "guanwu/incoming"):
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)
except Exception:
    # 만들어도 되고, 못 만들어도 서비스는 계속 되게 조용히 지나감
    pass


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
