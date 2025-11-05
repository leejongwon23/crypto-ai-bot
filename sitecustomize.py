# === sitecustomize.py (YOPO v1.3 — 경로자동변환 강화판) ===
# 하는 일:
# 1) 코드 어디서든 "/persistent/..." 쓰면 실제로는 BASE로 보냄
# 2) BASE 하위 폴더 미리 만들어둠
# 3) os/open/shutil/rename/replace 전부 패치
# 4) pathlib.Path.open 도 패치
# 5) sqlite3.connect 도 패치
# 6) 자주 쓰는 CSV는 빈 파일이라도 미리 만들어둠

import os, builtins, os.path, shutil, sqlite3, pathlib

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
        "failure_db",  # sqlite가 여기 쓸 수도 있으니까
    ):
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)
except Exception:
    pass


def _fix(path):
    """모든 경로에서 /persistent → BASE로 자동 변환"""
    if isinstance(path, pathlib.Path):
        path = str(path)
    if isinstance(path, str) and path.startswith("/persistent"):
        return path.replace("/persistent", BASE, 1)
    return path


# === 원본 함수 백업 ===
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

# pathlib / sqlite 원본
_OrigPathOpen = pathlib.Path.open
_orig_sqlite_connect = sqlite3.connect


def _ensure_parent(path: str):
    """/persistent/... 를 BASE로 바꾼 뒤 부모 폴더가 없으면 만든다."""
    path = _fix(path)
    parent = os.path.dirname(path)
    if parent and not _orig_exists(parent):
        try:
            _orig_makedirs(parent, exist_ok=True)
        except Exception:
            pass
    return path


# === 패치 함수들 ===
def open_patched(path, *a, **kw):
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
    if _orig_replace is None:
        raise AttributeError("os.replace not available")
    dst_fixed = _ensure_parent(dst)
    return _orig_replace(_fix(src), dst_fixed, *a, **kw)

def rename_patched(src, dst, *a, **kw):
    if _orig_rename is None:
        raise AttributeError("os.rename not available")
    dst_fixed = _ensure_parent(dst)
    return _orig_rename(_fix(src), dst_fixed, *a, **kw)


# === pathlib.Path.open 패치 (중요) ===
def path_open_patched(self, *a, **kw):
    fixed = _fix(str(self))
    mode = kw.get("mode") or (a[0] if a else "r")
    if any(m in mode for m in ("w", "a", "x", "+")):
        _ensure_parent(fixed)
    # pathlib.Path.open 원래 기능 대신 우리가 고친 경로로 열기
    return _orig_open(fixed, *a, **kw)

pathlib.Path.open = path_open_patched


# === sqlite3.connect 패치 (중요) ===
def sqlite_connect_patched(db, *a, **kw):
    # 메모리 DB나 상대경로는 건드리지 말기
    if isinstance(db, str) and db.startswith("/persistent"):
        db = _ensure_parent(db)  # 디렉터리 보장 + /persistent → BASE
    return _orig_sqlite_connect(db, *a, **kw)

sqlite3.connect = sqlite_connect_patched


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
        if not _orig_exists(fpath):
            _ensure_parent(fpath)
            with _orig_open(fpath, "w", encoding="utf-8-sig") as wf:
                wf.write("")
    except Exception:
        pass

print(f"[sitecustomize] 경로자동변환 활성화됨 → BASE={BASE}")
