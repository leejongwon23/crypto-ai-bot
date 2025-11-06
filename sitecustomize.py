# === sitecustomize.py (YOPO v1.7 — /persistent 전면 가로채기 + 중복실행 안전 + fd 안전모드) ===
# "/persistent/..." 로 접근하는 걸 전부 실제 저장소(BASE)로 보냄
# gunicorn 같은 서버가 fd(숫자)로 open 해도 안 터지게 함
# 워커가 import를 다시 해도 죽이지 않고 조용히 넘어감

import os, builtins, os.path, shutil, sqlite3, pathlib, io

loaded = os.environ.get("_YOPO_SITE_CUSTOMIZE_LOADED") == "1"

# ✅ 실제로 저장할 루트
BASE = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or "/tmp/persistent"
)
os.environ["PERSIST_DIR"] = BASE

if not loaded:
    # 첫 번째 import일 때만 디렉터리 만들고 패치 적용
    os.environ["_YOPO_SITE_CUSTOMIZE_LOADED"] = "1"

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
            "failure_db",
        ):
            os.makedirs(os.path.join(BASE, sub), exist_ok=True)
    except Exception:
        pass

    def _fix(path):
        # 숫자 fd, None, 이상한 타입은 그대로 둔다
        if isinstance(path, (int, float)) or path is None:
            return path
        if isinstance(path, pathlib.Path):
            path = str(path)
        if not isinstance(path, str):
            return path
        if path.startswith("/persistent"):
            return path.replace("/persistent", BASE, 1)
        return path

    _orig_open          = builtins.open
    _orig_io_open       = io.open
    _orig_exists        = os.path.exists
    _orig_isdir         = os.path.isdir
    _orig_makedirs      = os.makedirs
    _orig_listdir       = os.listdir
    _orig_move          = shutil.move
    _orig_remove        = os.remove
    _orig_rmdir         = os.rmdir
    _orig_rmtree        = shutil.rmtree
    _orig_replace       = getattr(os, "replace", None)
    _orig_rename        = getattr(os, "rename", None)
    _OrigPathOpen        = pathlib.Path.open
    _orig_sqlite_connect = sqlite3.connect

    def _ensure_parent(path):
        path = _fix(path)
        if not isinstance(path, str):
            return path
        parent = os.path.dirname(path)
        if parent and not _orig_exists(parent):
            try:
                _orig_makedirs(parent, exist_ok=True)
            except Exception:
                pass
        return path

    def open_patched(path, *a, **kw):
        if isinstance(path, (int, float)) or path is None:
            return _orig_open(path, *a, **kw)
        mode = kw.get("mode") or (a[0] if a else "r")
        fixed = _fix(path)
        if any(m in mode for m in ("w", "a", "x", "+")) and isinstance(fixed, str):
            _ensure_parent(fixed)
        return _orig_open(fixed, *a, **kw)

    def io_open_patched(path, *a, **kw):
        if isinstance(path, (int, float)) or path is None:
            return _orig_io_open(path, *a, **kw)
        mode = kw.get("mode") or (a[0] if a else "r")
        fixed = _fix(path)
        if any(m in mode for m in ("w", "a", "x", "+")) and isinstance(fixed, str):
            _ensure_parent(fixed)
        return _orig_io_open(fixed, *a, **kw)

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

    def path_open_patched(self, *a, **kw):
        fixed = _fix(str(self))
        mode = kw.get("mode") or (a[0] if a else "r")
        if any(m in mode for m in ("w", "a", "x", "+")) and isinstance(fixed, str):
            _ensure_parent(fixed)
        return _orig_open(fixed, *a, **kw)
    pathlib.Path.open = path_open_patched

    def sqlite_connect_patched(db, *a, **kw):
        if isinstance(db, str) and db.startswith("/persistent"):
            db = _ensure_parent(db)
            db = _fix(db)
        return _orig_sqlite_connect(db, *a, **kw)
    sqlite3.connect = sqlite_connect_patched

    builtins.open   = open_patched
    io.open         = io_open_patched
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

    for fname in ("wrong_predictions.csv", "prediction_log.csv", "train_log.csv"):
        fpath = os.path.join(BASE, fname)
        try:
            if not _orig_exists(fpath):
                _ensure_parent(fpath)
                with _orig_open(fpath, "w", encoding="utf-8-sig") as wf:
                    wf.write("")
        except Exception:
            pass

    print(f"[sitecustomize] 경로자동변환 활성화됨 → BASE={BASE} PERSIST_DIR={os.getenv('PERSIST_DIR')} cwd={os.getcwd()}")
else:
    # 두 번째 import부터는 여기만 찍힘
    print("[sitecustomize] already loaded (worker)")
