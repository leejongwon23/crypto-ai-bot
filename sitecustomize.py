# sitecustomize.py
import os, builtins, os.path, shutil

# 네가 Render에서 환경변수로 넣어둔 값 쓰기
BASE = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/tmp/persistent"
)

def _fix(path):
    # 문자열이고 /persistent로 시작하면 우리가 지정한 경로로 교체
    if isinstance(path, str) and path.startswith("/persistent"):
        return path.replace("/persistent", BASE, 1)
    return path

# 원래 함수들
_orig_open = builtins.open
_orig_exists = os.path.exists
_orig_isdir = os.path.isdir
_orig_makedirs = os.makedirs
_orig_listdir = os.listdir
_orig_move = shutil.move

# 가로채기
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

# 실제로 덮어쓰기
builtins.open = open_patched
os.path.exists = exists_patched
os.path.isdir = isdir_patched
os.makedirs = makedirs_patched
os.listdir = listdir_patched
shutil.move = move_patched
