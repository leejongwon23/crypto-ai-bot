# === sitecustomize.py (YOPO v1.2 — 경로자동변환 + 기본폴더/기본파일 보장 + replace/rename 패치) ===
# 역할:
# 1) 코드 어디서든 "/persistent/..." 를 쓰면 실제로는 BASE로 보냄
# 2) BASE랑 우리가 자주 쓰는 하위 폴더를 미리 만들어둠
# 3) os.replace / os.rename 까지 패치해서 JSON 저장도 안 터지게 함
# 4) 초기화 시 자주 참조하는 CSV는 빈 파일이라도 만들어둠

import os, builtins, os.path, shutil

# ✅ 실제로 저장할 루트
BASE = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/tmp/persistent"   # 최후 기본값
)

# ✅ 1) 루트랑 자주 쓰는 디렉터리 먼저 만들어두기
try:
    os.makedirs(BASE, exist_ok=True)
    for sub in (
        "logs",
        "state",
        "locks",
        "importances",
        "guanwu/incoming",
        "ssl_models",
        "models",
        "run",
    ):
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)
except Exception:
    # 여기서 죽어도 전체 앱은 계속 가야 함
    pass


def _fix(path):
    """모든 경로에서 /persistent → BASE로 자동 변환"""
    import pathlib
    if isinstance(path, pathlib.Path):
        path = str(path)
    if isinstance(path, str) and path.startswith("/persistent"):
        return path.replace("/persistent", BASE, 1)
    return path


# ✅ 원본 함수 백업
_orig_open      = builtins.open
_orig_exists    = os.path.exists
_orig_isdir     = os.path.isdir
_orig_makedirs  = os.makedirs
_orig_listdir   = os.listdir
_orig_move      = shutil.move
_orig_remove    = os.remove
_orig_rmdir     = os.rmdir
_orig_rmtree    = shutil.rmtree
_orig_replace   = getattr(os, "replace", None)
_orig_rename    = getattr(os, "rename", None)


def _ensure_parent(path: str):
    """/persistent/... 를 BASE로 바꾼 뒤 부모 폴더가 없으면 만든다."""
    path = _fix(path)
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
    return path


# === 패치 함수들 ===
def open_patched(path, *a, **kw):
    # 쓰기 모드면 부모 디렉터리 먼저 만든다
    mode = kw.get("mode") or (a[0] if a else "r")
    fixed = _fix(path)
    if any(m in mode for m in ("w", "a", "x", "+")):
        _ensure_parent(fixed)
    return _orig_open(fixed, *a, **kw)

def exists_patched(path):
    return _orig_exists(_fix(path))

def isdir_patched(path):
    return _orig_isdir(_fix(path))

def makedirs_patched(path, *a, **kw):
    return _orig_makedirs(_fix(path), *a, **kw)

def listdir_patched(path):
    return _orig_listdir(_fix(path))

def move_patched(src, dst, *a, **kw):
    dst_fixed = _ensure_parent(dst)
    return _orig_move(_fix(src), dst_fixed, *a, **kw)

def remove_patched(path, *a, **kw):
    return _orig_remove(_fix(path), *a, **kw)

def rmdir_patched(path, *a, **kw):
    return _orig_rmdir(_fix(path), *a, **kw)

def rmtree_patched(path, *a, **kw):
    return _orig_rmtree(_fix(path), *a, **kw)

def replace_patched(src, dst, *a, **kw):
    # config.py 같은 데서 tmp -> real 바꿀 때 여기로 들어옴
    if _orig_replace is None:
        raise AttributeError("os.replace not available")
    dst_fixed = _ensure_parent(dst)
    return _orig_replace(_fix(src), dst_fixed, *a, **kw)

def rename_patched(src, dst, *a, **kw):
    if _orig_rename is None:
        raise AttributeError("os.rename not available")
    dst_fixed = _ensure_parent(dst)
    return _orig_rename(_fix(src), dst_fixed, *a, **kw)


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
if _orig_replace is not None:
    os.replace  = replace_patched
if _orig_rename is not None:
    os.rename   = rename_patched


# ✅ 2) 자주 터지는 CSV는 여기서 미리 "빈 파일"이라도 만들어 놓자
for fname in (
    "wrong_predictions.csv",
    "prediction_log.csv",
    "train_log.csv",
):
    fpath = os.path.join(BASE, fname)
    try:
        if not os.path.exists(fpath):
            # 헤더는 logger가 나중에 다시 보정하니까 지금은 그냥 빈 파일만
            _ensure_parent(fpath)
            with open(fpath, "w", encoding="utf-8-sig") as wf:
                wf.write("")
    except Exception:
        pass

print(f"[sitecustomize] 경로자동변환 활성화됨 → BASE={BASE}")
