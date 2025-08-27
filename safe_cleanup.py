# safe_cleanup.py (FIXED-CONFIG: env 없이 동작, 스케줄러 포함 / micro-fix3, 10GB 서버용 튜닝 + 모델·메타 세트 정리 강화)
import os
import time
import threading
import gc
from datetime import datetime, timedelta

# ====== 기본 경로 (env 있으면 사용, 없으면 기본) ======
ROOT_DIR = os.getenv("PERSIST_ROOT", "/persistent")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
SSL_DIR = os.path.join(ROOT_DIR, "ssl_models")
LOCK_DIR = os.path.join(ROOT_DIR, "locks")
DELETED_LOG_PATH = os.path.join(LOG_DIR, "deleted_log.txt")

# ====== 정책(고정값 / 10GB 환경 최적화) ======
KEEP_DAYS   = 1
HARD_CAP_GB = 9.6   # 10GB 한계 대비 여유
SOFT_CAP_GB = 9.0
TRIGGER_GB  = 7.5   # 여유 확보를 위해 약간 상향(7.0→7.5)
MIN_FREE_GB = 0.8   # 하드캡 해제 후 최소 확보 목표

CSV_MAX_MB = 50
CSV_BACKUPS = 3

MAX_MODELS_KEEP_GLOBAL = 200
MAX_MODELS_PER_KEY = 2

PROTECT_HOURS = 12
LOCK_PATH = os.path.join(LOCK_DIR, "train_or_predict.lock")
DRYRUN = False

# ✅ (5번) 압축 모델 확장자도 동일 취급
MODEL_EXTS = (".pt", ".ptz", ".safetensors")
META_EXT = ".meta.json"
_PREFERRED_WEIGHT_EXTS = (".ptz", ".safetensors", ".pt")  # 보존 우선순위

DELETE_PREFIXES = ["prediction_", "evaluation_", "wrong_", "model_", "ssl_", "meta_", "evo_"]
EXCLUDE_FILES = {
    "prediction_log.csv", "train_log.csv", "evaluation_result.csv",
    "deleted_log.txt", "wrong_predictions.csv", "fine_tune_target.csv"
}

ROOT_CSVS = [
    os.path.join(ROOT_DIR, "prediction_log.csv"),
    os.path.join(ROOT_DIR, "wrong_predictions.csv"),
    os.path.join(ROOT_DIR, "evaluation_result.csv"),
    os.path.join(ROOT_DIR, "train_log.csv"),
]

# ----------------- 공통 유틸 -----------------
def _size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return 0

def get_directory_size_gb(path):
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += _size_bytes(fp)
    return total / (1024 ** 3)

def _human_gb(v): return f"{v:.2f}GB"

def _list_files(dir_path):
    try:
        return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    except Exception:
        return []

def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SSL_DIR, exist_ok=True)
    os.makedirs(LOCK_DIR, exist_ok=True)

def _is_within(child: str, parent: str) -> bool:
    try:
        child_abs  = os.path.abspath(child)
        parent_abs = os.path.abspath(parent)
        return os.path.commonpath([child_abs, parent_abs]) == parent_abs
    except Exception:
        return False

def _is_model_file(path: str) -> bool:
    """models/ 내부의 .pt/.ptz/.safetensors 및 짝 메타를 모델로 본다."""
    if not isinstance(path, str):
        return False
    base = os.path.basename(path)
    # 모델 가중치
    if any(base.endswith(ext) for ext in MODEL_EXTS):
        return True
    # 메타(모델과 세트)
    if base.endswith(META_EXT):
        return True
    return False

def _should_delete_file(fname: str) -> bool:
    """
    기존 규칙 + (NEW) models/ 안의 모델 확장자는 접두사 없이도 정리 대상으로 인정.
    """
    base = os.path.basename(fname)
    # 보호 목록
    if base in EXCLUDE_FILES:
        return False
    # models/ 디렉토리의 모델/메타 파일은 접두사와 무관하게 삭제 후보
    try:
        if _is_within(fname, MODEL_DIR):
            if _is_model_file(fname):
                return True
    except Exception:
        pass
    # 일반 접두사 규칙
    return any(base.startswith(p) for p in DELETE_PREFIXES)

def _is_recent(path: str, hours: float) -> bool:
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) < timedelta(hours=hours)
    except Exception:
        return False

def _rollover_csv(path: str, max_mb: int, backups: int):
    if not os.path.isfile(path):
        return []
    size_mb = _size_bytes(path) / (1024 ** 2)
    if size_mb <= max_mb:
        return []
    deleted = []
    for i in range(backups, 0, -1):
        bak = f"{path}.{i}"
        older = f"{path}.{i+1}"
        if os.path.exists(older):
            if not DRYRUN:
                os.remove(older)
            deleted.append(older)
        if os.path.exists(bak):
            if not DRYRUN:
                os.rename(bak, older)
    if not DRYRUN:
        os.rename(path, f"{path}.1")
        open(path, "w", encoding="utf-8").close()
    return deleted

def _delete_file(path: str, deleted_log: list):
    try:
        if DRYRUN:
            print(f"[DRYRUN] 삭제 예정: {path}")
            return
        os.remove(path)
        deleted_log.append(path)
        print(f"[🗑 삭제] {path}")
    except Exception as e:
        print(f"[경고] 삭제 실패: {path} | {e}")

# ----------------- 모델·메타 세트 관리 -----------------
def _split_stem_and_ext(path: str):
    """
    returns: (stem, ext)
      - meta: "xxx.meta.json" -> ("xxx", ".meta.json")
      - weight: "xxx.ptz" -> ("xxx", ".ptz")
    """
    base = os.path.basename(path)
    if base.endswith(".meta.json"):
        stem = base[:-10]  # drop ".meta.json"
        ext = ".meta.json"
        return stem, ext
    root, ext = os.path.splitext(base)
    return root, ext

def _collect_model_sets():
    """
    models/ 내 파일들을 stem 기준으로 세트(가중치+메타)로 묶어서 반환.
    return: dict[stem] = {"weights": {ext: path}, "meta": path|None, "mtime": latest_mtime}
    (보호시간에 해당하는 최신 파일은 제외하여 삭제 후보만 대상으로 함)
    """
    sets = {}
    for p in _list_files(MODEL_DIR):
        if not os.path.isfile(p):
            continue
        if not _is_model_file(p):
            continue
        if _is_recent(p, PROTECT_HOURS):
            continue
        stem, ext = _split_stem_and_ext(p)
        st = sets.setdefault(stem, {"weights": {}, "meta": None, "mtime": 0.0})
        try:
            mtime = os.path.getmtime(p)
        except Exception:
            mtime = 0.0
        st["mtime"] = max(st["mtime"], mtime)
        if ext == ".meta.json":
            st["meta"] = p
        else:
            st["weights"][ext] = p
    return sets

def _key_from_stem(stem: str) -> str:
    """
    stem: 'BTCUSDT_단기_lstm_group1_cls3' -> key: 'BTCUSDT_단기_lstm'
    (심볼_전략_모델 단위로 버킷팅)
    """
    parts = stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else stem

# ----------------- 삭제/보존 로직 -----------------
def _delete_old_by_days(paths, cutoff_dt, deleted_log, accept_all=False):
    for d in paths:
        for p in _list_files(d):
            if not os.path.isfile(p):
                continue
            if not accept_all and not _should_delete_file(p):
                continue
            if _is_recent(p, PROTECT_HOURS):
                continue
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(p))
            except Exception:
                continue
            if mtime < cutoff_dt:
                _delete_file(p, deleted_log)

def _delete_until_target(deleted_log, target_gb):
    candidates = []
    # LOG/MODEL: 규칙 기반 후보 수집
    for d in [LOG_DIR, MODEL_DIR]:
        for p in _list_files(d):
            if os.path.isfile(p) and _should_delete_file(p):
                if _is_recent(p, PROTECT_HOURS):
                    continue
                try:
                    ctime = os.path.getctime(p)
                except Exception:
                    ctime = 0
                candidates.append((ctime, p))
    # SSL: 대용량 캐시 우선 제거
    for p in _list_files(SSL_DIR):
        if os.path.isfile(p) and not _is_recent(p, PROTECT_HOURS):
            try:
                ctime = os.path.getctime(p)
            except Exception:
                ctime = 0
            candidates.append((ctime, p))

    # 오래된 것부터
    candidates.sort(key=lambda x: x[0])
    while get_directory_size_gb(ROOT_DIR) > target_gb and candidates:
        _, p = candidates.pop(0)
        _delete_file(p, deleted_log)

def _limit_models_per_key(deleted_log):
    """
    모델/메타를 '세트'로 묶어 심볼_전략_모델 단위로 MAX_MODELS_PER_KEY '세트'만 남긴다.
    - 세트 내 가중치는 선호 확장자 1개만 유지(.ptz > .safetensors > .pt)
    - 세트 외 나머지 가중치/메타는 삭제
    - 전체 상한(MAX_MODELS_KEEP_GLOBAL)도 세트 기준으로 적용
    """
    sets = _collect_model_sets()
    if not sets:
        return

    # 세트 단위로 정렬(최신 mtime 우선)
    items = [(stem, data) for stem, data in sets.items()]
    items.sort(key=lambda x: x[1]["mtime"], reverse=True)

    # 글로벌 상한: 초과 세트는 전부 삭제
    if len(items) > MAX_MODELS_KEEP_GLOBAL:
        for stem, data in items[MAX_MODELS_KEEP_GLOBAL:]:
            for wpath in data["weights"].values():
                _delete_file(wpath, deleted_log)
            if data["meta"]:
                _delete_file(data["meta"], deleted_log)
        items = items[:MAX_MODELS_KEEP_GLOBAL]

    # 버킷(심볼_전략_모델)별 상한 적용
    from collections import defaultdict
    buckets = defaultdict(list)
    for stem, data in items:
        buckets[_key_from_stem(stem)].append((stem, data))

    for key, arr in buckets.items():
        # 최신 세트 MAX_MODELS_PER_KEY개만 보존, 나머지는 삭제
        arr.sort(key=lambda x: x[1]["mtime"], reverse=True)
        keep = arr[:MAX_MODELS_PER_KEY]
        drop = arr[MAX_MODELS_PER_KEY:]

        # 보존 세트: 가중치는 선호 확장자 1개만 유지(+메타는 그대로 유지)
        for stem, data in keep:
            chosen = None
            for ext in _PREFERRED_WEIGHT_EXTS:
                if ext in data["weights"]:
                    chosen = ext
                    break
            for ext, wpath in list(data["weights"].items()):
                if chosen is not None and ext == chosen:
                    continue
                _delete_file(wpath, deleted_log)
            # 메타는 있으면 보존 (삭제하지 않음)

        # 드롭 세트: 전부 삭제(가중치+메타)
        for stem, data in drop:
            for wpath in data["weights"].values():
                _delete_file(wpath, deleted_log)
            if data["meta"]:
                _delete_file(data["meta"], deleted_log)

def _vacuum_sqlite():
    targets = []
    for base in [ROOT_DIR, LOG_DIR]:
        for f in _list_files(base):
            if os.path.isfile(f) and f.lower().endswith(".db"):
                targets.append(f)
    for path in targets:
        try:
            import sqlite3
            con = sqlite3.connect(path)
            con.execute("VACUUM;")
            con.close()
            print(f"[VACUUM] {os.path.basename(path)} 완료")
        except Exception as e:
            print(f"[경고] VACUUM 실패: {path} | {e}")

def _locked_by_runtime() -> bool:
    if os.path.exists(LOCK_PATH):
        print(f"[⛔ 중단] LOCK 발견: {LOCK_PATH}")
        return True
    try:
        for f in _list_files(LOCK_DIR):
            if f.endswith(".lock"):
                print(f"[⛔ 중단] LOCK 발견: {f}")
                return True
    except Exception:
        pass
    return False

# ========= 🆘 EMERGENCY PURGE (접두사/보호시간/락 무시) =========
def emergency_purge(target_gb=None):
    """
    디스크가 꽉 찼을 때 즉시 용량 확보.
    - 접두사/보호시간/락 조건 무시
    - ssl_models → models → logs 순서
    - 오래된 파일부터 삭제
    - target_gb 미지정 시: max(SOFT_CAP_GB, HARD_CAP_GB - MIN_FREE_GB)
    """
    _ensure_dirs()
    deleted = []
    target = target_gb or max(SOFT_CAP_GB, HARD_CAP_GB - MIN_FREE_GB)

    def _collect_all(dirpath):
        items = []
        for p in _list_files(dirpath):
            if not os.path.isfile(p):
                continue
            if os.path.basename(p) == "deleted_log.txt":
                continue
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0
            items.append((mtime, p))
        items.sort(key=lambda x: x[0])  # 오래된 것 먼저
        return [p for _, p in items]

    print("[🆘 EMERGENCY] 즉시 강제 정리 시작 (락/보호시간 무시)")
    ordered_dirs = [SSL_DIR, MODEL_DIR, LOG_DIR]
    candidates = []
    for d in ordered_dirs:
        candidates.extend(_collect_all(d))

    while get_directory_size_gb(ROOT_DIR) > target and candidates:
        p = candidates.pop(0)
        _delete_file(p, deleted)

    _vacuum_sqlite()

    if deleted:
        now = datetime.now()
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(DELETED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [EMERGENCY] 삭제 파일:\n")
                for path in deleted:
                    f.write(f"  - {path}\n")
            print(f"[🆘 EMERGENCY] 총 {len(deleted)}개 파일 삭제")
        except Exception as e:
            print(f"[⚠️ EMERGENCY 로그 기록 실패] → {e}")
            print(f"[🆘 EMERGENCY] 총 {len(deleted)}개 파일 삭제(로그 기록 생략)")

def run_emergency_purge():
    """앱에서 한 줄로 호출하기 위한 래퍼"""
    emergency_purge()

# ========= 일반 주기 정리 =========
def auto_delete_old_logs():
    _ensure_dirs()

    if _locked_by_runtime():
        return

    now = datetime.now()
    cutoff = now - timedelta(days=KEEP_DAYS)
    deleted = []

    current_gb = get_directory_size_gb(ROOT_DIR)
    print(f"[용량] 현재={_human_gb(current_gb)} | 트리거={_human_gb(TRIGGER_GB)} | 목표={_human_gb(SOFT_CAP_GB)} | 하드캡={_human_gb(HARD_CAP_GB)}")

    # CSV 롤오버(먼저 공간 조금 확보)
    for csv_path in ROOT_CSVS + [os.path.join(LOG_DIR, n) for n in ["prediction_log.csv", "train_log.csv", "evaluation_result.csv", "wrong_predictions.csv"]]:
        deleted += _rollover_csv(csv_path, CSV_MAX_MB, CSV_BACKUPS)

    if current_gb >= HARD_CAP_GB:
        print(f"[🚨 하드캡 초과] 즉시 강제 정리 시작")
        # 1) SSL(대용량) → 2) 모델/로그 순
        _delete_old_by_days([SSL_DIR],  cutoff, deleted_log=deleted, accept_all=True)
        _delete_old_by_days([MODEL_DIR, LOG_DIR], cutoff, deleted_log=deleted)
        _delete_until_target(deleted, max(SOFT_CAP_GB, HARD_CAP_GB - MIN_FREE_GB))
        _limit_models_per_key(deleted)
        _vacuum_sqlite()

    elif current_gb >= TRIGGER_GB:
        print(f"[⚠️ 트리거 초과] 정리 시작")
        _delete_old_by_days([SSL_DIR],  cutoff, deleted_log=deleted, accept_all=True)
        _delete_old_by_days([MODEL_DIR, LOG_DIR], cutoff, deleted_log=deleted)
        _delete_until_target(deleted, SOFT_CAP_GB)
        _limit_models_per_key(deleted)
        _vacuum_sqlite()
    else:
        print(f"[✅ 용량정상] 정리 불필요")
        _limit_models_per_key(deleted)

    if deleted:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(DELETED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 삭제된 파일 목록:\n")
                for path in deleted:
                    f.write(f"  - {path}\n")
            print(f"[🧹 삭제 완료] 총 {len(deleted)}개 파일 정리")
        except Exception as e:
            print(f"[⚠️ 삭제 로그 기록 실패] → {e}")
            print(f"[🧹 삭제 완료] 총 {len(deleted)}개 파일 정리(로그 기록 생략)")

def cleanup_logs_and_models():
    auto_delete_old_logs()

# ====== 경량/주기 실행 유틸(고정값) ======
INTERVAL_SEC = 300
RUN_ON_START = True
_VERBOSE = True

def _log(msg: str):
    if _VERBOSE:
        print(f"[safe_cleanup] {msg}")

def _light_cleanup():
    try:
        from cache import CacheManager
        try:
            before = CacheManager.stats()
        except Exception:
            before = None
        pruned = CacheManager.prune()
        try:
            after = CacheManager.stats()
        except Exception:
            after = None
        _log(f"cache prune ok: before={before}, after={after}, pruned={pruned}")
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

def start_cleanup_scheduler(daemon: bool = True) -> threading.Thread:
    def _loop():
        if RUN_ON_START:
            _log("초기 1회 실행")
            auto_delete_old_logs()
        while True:
            time.sleep(INTERVAL_SEC)
            _log("주기 실행")
            auto_delete_old_logs()
    t = threading.Thread(target=_loop, name="safe-cleanup-scheduler", daemon=daemon)
    t.start()
    _log(f"스케줄러 시작(주기 {INTERVAL_SEC}s, daemon={daemon})")
    return t

def trigger_light_cleanup():
    _light_cleanup()

if __name__ == "__main__":
    start_cleanup_scheduler(daemon=False)
