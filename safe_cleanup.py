# safe_cleanup.py
import os
import shutil
from datetime import datetime, timedelta

# ====== 기본 경로 ======
ROOT_DIR = os.getenv("PERSIST_ROOT", "/persistent")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DELETED_LOG_PATH = os.path.join(LOG_DIR, "deleted_log.txt")

# ====== 정책(환경변수로 조정 가능) ======
KEEP_DAYS = int(os.getenv("SAFE_KEEP_DAYS", "1"))                # 며칠 지난 파일부터 우선 삭제
HARD_CAP_GB = float(os.getenv("SAFE_HARD_CAP_GB", "9.8"))        # 절대 상한선(넘으면 강제 정리)
TRIGGER_GB = float(os.getenv("SAFE_TRIGGER_GB", "7.0"))          # 이 값 이상이면 정리 시작
SOFT_CAP_GB = float(os.getenv("SAFE_SOFT_CAP_GB", "9.2"))        # 정리 목표치(여기까지 낮추기)
MIN_FREE_GB = float(os.getenv("SAFE_MIN_FREE_GB", "1.0"))        # 최소 여유공간 확보 목표

CSV_MAX_MB = int(os.getenv("SAFE_CSV_MAX_MB", "50"))             # 초과하면 롤오버
CSV_BACKUPS = int(os.getenv("SAFE_CSV_BACKUPS", "3"))            # 롤오버 보관 개수

MAX_MODELS_KEEP_GLOBAL = int(os.getenv("SAFE_MAX_MODELS_KEEP_GLOBAL", "200"))  # models 전체 상한
MAX_MODELS_PER_KEY = int(os.getenv("SAFE_MAX_MODELS_PER_KEY", "2"))           # 같은 키(심볼/전략) 보관 개수

DRYRUN = os.getenv("SAFE_CLEANUP_DRYRUN", "0") == "1"            # 삭제 시도만 로그(실제 삭제 X)

# 삭제 후보 접두사 / 제외 파일
DELETE_PREFIXES = ["prediction_", "evaluation_", "wrong_", "model_", "ssl_", "meta_", "evo_"]
EXCLUDE_FILES = {
    "prediction_log.csv", "train_log.csv", "evaluation_result.csv",
    "deleted_log.txt", "wrong_predictions.csv", "fine_tune_target.csv"
}

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
    # 오래된 백업부터 제거
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
    # 현재 파일 → .1 로 회전
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

def _delete_old_by_days(paths, cutoff_dt, deleted_log):
    for d in paths:
        for p in _list_files(d):
            if not os.path.isfile(p):
                continue
            if not _should_delete_file(p):
                continue
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(p))
            except Exception:
                continue
            if mtime < cutoff_dt:
                _delete_file(p, deleted_log)

def _delete_until_target(deleted_log, target_gb):
    # 오래된 순으로 삭제
    candidates = []
    for d in [LOG_DIR, MODEL_DIR]:
        for p in _list_files(d):
            if os.path.isfile(p) and _should_delete_file(p):
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
    # 파일명 규칙 예: SYMBOL_전략_모델타입_groupX_clsY.pt / .meta.json
    files = [p for p in _list_files(MODEL_DIR) if os.path.isfile(p)]
    # 전체 상한 먼저
    files.sort(key=os.path.getmtime, reverse=True)
    if len(files) > MAX_MODELS_KEEP_GLOBAL:
        for p in files[MAX_MODELS_KEEP_GLOBAL:]:
            _delete_file(p, deleted_log)

    # 키별 상한
    from collections import defaultdict
    buckets = defaultdict(list)
    for p in files:
        base = os.path.basename(p)
        # 키 추출(확장자 제거)
        key = base.split(".")[0]
        # 심플 키(심볼_전략_모델명)만 남기고 group/cls는 같이 묶임
        parts = key.split("_")
        if len(parts) >= 3:
            simple = "_".join(parts[:3])
        else:
            simple = key
        buckets[simple].append(p)

    for key, items in buckets.items():
        items.sort(key=os.path.getmtime, reverse=True)
        for p in items[MAX_MODELS_PER_KEY:]:
            _delete_file(p, deleted_log)

def _vacuum_sqlite():
    # sqlite 파일이 있다면 VACUUM (공간 회수)
    for fname in ["success_log.db", "failure_log.db"]:
        path = os.path.join(ROOT_DIR, fname)
        if not os.path.isfile(path):
            continue
        try:
            import sqlite3
            con = sqlite3.connect(path)
            con.execute("VACUUM;")
            con.close()
            print(f"[VACUUM] {fname} 완료")
        except Exception as e:
            print(f"[경고] VACUUM 실패: {fname} | {e}")

def auto_delete_old_logs():
    _ensure_dirs()
    now = datetime.now()
    cutoff = now - timedelta(days=KEEP_DAYS)
    deleted = []

    current_gb = get_directory_size_gb(ROOT_DIR)
    print(f"[용량] 현재={_human_gb(current_gb)} | 트리거={_human_gb(TRIGGER_GB)} | 목표={_human_gb(SOFT_CAP_GB)} | 하드캡={_human_gb(HARD_CAP_GB)}")

    # CSV 로그 롤오버(너무 큰 CSV는 회전)
    for csv_name in ["prediction_log.csv", "train_log.csv", "evaluation_result.csv", "wrong_predictions.csv"]:
        csv_path = os.path.join(LOG_DIR, csv_name)
        deleted += _rollover_csv(csv_path, CSV_MAX_MB, CSV_BACKUPS)

    # 하드캡 초과면 즉시 강제 정리
    if current_gb >= HARD_CAP_GB:
        print(f"[🚨 하드캡 초과] 즉시 강제 정리 시작")
        _delete_old_by_days([LOG_DIR, MODEL_DIR], cutoff, deleted)
        _delete_until_target(deleted, max(SOFT_CAP_GB, HARD_CAP_GB - MIN_FREE_GB))
        _limit_models_per_key(deleted)
        _vacuum_sqlite()

    # 트리거 초과면 일반 정리
    elif current_gb >= TRIGGER_GB:
        print(f"[⚠️ 트리거 초과] 정리 시작")
        _delete_old_by_days([LOG_DIR, MODEL_DIR], cutoff, deleted)
        _delete_until_target(deleted, SOFT_CAP_GB)
        _limit_models_per_key(deleted)
        _vacuum_sqlite()
    else:
        print(f"[✅ 용량정상] 정리 불필요")
        _limit_models_per_key(deleted)  # 정상이어도 모델 상한 유지

    # 삭제 로그 기록
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

# 앱에서 불러쓰는 래퍼
def cleanup_logs_and_models():
    auto_delete_old_logs()
