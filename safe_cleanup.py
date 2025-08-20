# safe_cleanup.py (FINAL)
import os
import shutil
from datetime import datetime, timedelta

# ====== 기본 경로 ======
ROOT_DIR = os.getenv("PERSIST_ROOT", "/persistent")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
SSL_DIR = os.path.join(ROOT_DIR, "ssl_models")
DELETED_LOG_PATH = os.path.join(LOG_DIR, "deleted_log.txt")

# ====== 정책(환경변수: SAFE_* 우선, DISK_* 백워드 호환) ======
KEEP_DAYS = int(os.getenv("SAFE_KEEP_DAYS", "1"))
HARD_CAP_GB = float(os.getenv("SAFE_HARD_CAP_GB", os.getenv("DISK_HARD_LIMIT_GB", "9.8")))
SOFT_CAP_GB = float(os.getenv("SAFE_SOFT_CAP_GB", os.getenv("DISK_SOFT_LIMIT_GB", "9.2")))
TRIGGER_GB  = float(os.getenv("SAFE_TRIGGER_GB", os.getenv("DISK_TRIGGER_GB", "7.0")))
MIN_FREE_GB = float(os.getenv("SAFE_MIN_FREE_GB", "1.0"))

CSV_MAX_MB = int(os.getenv("SAFE_CSV_MAX_MB", "50"))
CSV_BACKUPS = int(os.getenv("SAFE_CSV_BACKUPS", "3"))

MAX_MODELS_KEEP_GLOBAL = int(os.getenv("SAFE_MAX_MODELS_KEEP_GLOBAL", "200"))
MAX_MODELS_PER_KEY = int(os.getenv("SAFE_MAX_MODELS_PER_KEY", "2"))

DRYRUN = os.getenv("SAFE_CLEANUP_DRYRUN", "0") == "1"

# 삭제 후보 접두사 / 제외 파일
DELETE_PREFIXES = ["prediction_", "evaluation_", "wrong_", "model_", "ssl_", "meta_", "evo_"]
EXCLUDE_FILES = {
    "prediction_log.csv", "train_log.csv", "evaluation_result.csv",
    "deleted_log.txt", "wrong_predictions.csv", "fine_tune_target.csv"
}

# 루트 CSV (경로 일치 중요)
ROOT_CSVS = [
    os.path.join(ROOT_DIR, "prediction_log.csv"),
    os.path.join(ROOT_DIR, "wrong_predictions.csv"),
    os.path.join(ROOT_DIR, "evaluation_result.csv"),
    os.path.join(ROOT_DIR, "train_log.csv"),
]

def _size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return 0

def get_directory_size_gb(path):
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

def _should_delete_file(fname: str) -> bool:
    if os.path.basename(fname) in EXCLUDE_FILES:
        return False
    return any(os.path.basename(fname).startswith(p) for p in DELETE_PREFIXES)

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

def _delete_old_by_days(paths, cutoff_dt, deleted_log, accept_all=False):
    for d in paths:
        for p in _list_files(d):
            if not os.path.isfile(p):
                continue
            if not accept_all and not _should_delete_file(p):
                continue
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(p))
            except Exception:
                continue
            if mtime < cutoff_dt:
                _delete_file(p, deleted_log)

def _delete_until_target(deleted_log, target_gb):
    candidates = []
    for d in [LOG_DIR, MODEL_DIR]:
        for p in _list_files(d):
            if os.path.isfile(p) and _should_delete_file(p):
                try:
                    ctime = os.path.getctime(p)
                except Exception:
                    ctime = 0
                candidates.append((ctime, p))
    # SSL 모델은 접두사가 없으므로 모두 후보로
    for p in _list_files(SSL_DIR):
        if os.path.isfile(p):
            try:
                ctime = os.path.getctime(p)
            except Exception:
                ctime = 0
            candidates.append((ctime, p))

    candidates.sort(key=lambda x: x[0])  # 오래된 것부터
    while get_directory_size_gb(ROOT_DIR) > target_gb and candidates:
        _, p = candidates.pop(0)
        _delete_file(p, deleted_log)

def _limit_models_per_key(deleted_log):
    files = [p for p in _list_files(MODEL_DIR) if os.path.isfile(p)]
    files.sort(key=os.path.getmtime, reverse=True)
    if len(files) > MAX_MODELS_KEEP_GLOBAL:
        for p in files[MAX_MODELS_KEEP_GLOBAL:]:
            _delete_file(p, deleted_log)

    from collections import defaultdict
    buckets = defaultdict(list)
    for p in files:
        base = os.path.basename(p)
        key = base.split(".")[0]
        parts = key.split("_")
        simple = "_".join(parts[:3]) if len(parts) >= 3 else key  # SYMBOL_전략_모델타입
        buckets[simple].append(p)

    for key, items in buckets.items():
        items.sort(key=os.path.getmtime, reverse=True)
        for p in items[MAX_MODELS_PER_KEY:]:
            _delete_file(p, deleted_log)

def _vacuum_sqlite():
    # ROOT/LOG 디렉토리 내 모든 .db에 대해 VACUUM
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

def auto_delete_old_logs():
    _ensure_dirs()
    now = datetime.now()
    cutoff = now - timedelta(days=KEEP_DAYS)
    deleted = []

    current_gb = get_directory_size_gb(ROOT_DIR)
    print(f"[용량] 현재={_human_gb(current_gb)} | 트리거={_human_gb(TRIGGER_GB)} | 목표={_human_gb(SOFT_CAP_GB)} | 하드캡={_human_gb(HARD_CAP_GB)}")

    # ✅ CSV 롤오버 (루트 + LOG_DIR)
    for csv_path in ROOT_CSVS + [os.path.join(LOG_DIR, n) for n in ["prediction_log.csv", "train_log.csv", "evaluation_result.csv", "wrong_predictions.csv"]]:
        deleted += _rollover_csv(csv_path, CSV_MAX_MB, CSV_BACKUPS)

    if current_gb >= HARD_CAP_GB:
        print(f"[🚨 하드캡 초과] 즉시 강제 정리 시작")
        _delete_old_by_days([LOG_DIR, MODEL_DIR], cutoff, deleted_log=deleted)
        _delete_old_by_days([SSL_DIR], cutoff, deleted_log=deleted, accept_all=True)
        _delete_until_target(deleted, max(SOFT_CAP_GB, HARD_CAP_GB - MIN_FREE_GB))
        _limit_models_per_key(deleted)
        _vacuum_sqlite()

    elif current_gb >= TRIGGER_GB:
        print(f"[⚠️ 트리거 초과] 정리 시작")
        _delete_old_by_days([LOG_DIR, MODEL_DIR], cutoff, deleted_log=deleted)
        _delete_old_by_days([SSL_DIR], cutoff, deleted_log=deleted, accept_all=True)
        _delete_until_target(deleted, SOFT_CAP_GB)
        _limit_models_per_key(deleted)
        _vacuum_sqlite()
    else:
        print(f"[✅ 용량정상] 정리 불필요")
        _limit_models_per_key(deleted)

    if deleted:
        try:
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
