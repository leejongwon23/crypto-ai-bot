import os
import shutil
from datetime import datetime, timedelta

# ✅ 설정
LOG_DIR = "/persistent/logs"
MODEL_DIR = "/persistent/models"
DELETED_LOG_PATH = os.path.join(LOG_DIR, "deleted_log.txt")

KEEP_DAYS = 1           # ✅ 최근 1일만 유지
DISK_LIMIT_GB = 10      # 전체 허용 용량
TRIGGER_GB = 7          # 정리 시작 트리거

DELETE_PREFIXES = ["prediction_", "evaluation_", "wrong_"]
EXCLUDE_FILES = set([
    "prediction_log.csv", "train_log.csv", "evaluation_result.csv",
    "deleted_log.txt", "wrong_predictions.csv", "fine_tune_target.csv"
])

def get_directory_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)

def get_sorted_old_files(dir_path):
    files = []
    for fname in os.listdir(dir_path):
        if fname in EXCLUDE_FILES:
            continue
        if not any(fname.startswith(p) for p in DELETE_PREFIXES):
            continue
        full_path = os.path.join(dir_path, fname)
        if os.path.isfile(full_path):
            try:
                ctime = os.path.getctime(full_path)
                files.append((full_path, ctime))
            except: continue
    files.sort(key=lambda x: x[1])  # 오래된 순
    return files

def auto_delete_old_logs():
    now = datetime.now()
    cutoff = now - timedelta(days=KEEP_DAYS)
    deleted = []

    # ✅ 디스크 용량 확인
    current_gb = get_directory_size_gb("/persistent")
    if current_gb < TRIGGER_GB:
        print(f"[✅ 용량정상] 현재 사용량: {current_gb:.2f}GB → 정리 안함")
        return

    print(f"[⚠️ 용량초과] {current_gb:.2f}GB → 로그/모델 정리 시작")

    if not os.path.exists(LOG_DIR):
        print(f"[❌ 로그 디렉토리 없음] {LOG_DIR}")
        return

    # ✅ 오래된 파일 삭제 우선
    for fname in os.listdir(LOG_DIR):
        fpath = os.path.join(LOG_DIR, fname)
        if not os.path.isfile(fpath): continue
        if fname in EXCLUDE_FILES: continue
        if not any(fname.startswith(p) for p in DELETE_PREFIXES): continue

        try:
            date_str = fname.split("_")[-1].replace(".csv", "").strip()
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                os.remove(fpath)
                deleted.append(fname)
        except: continue

    # ✅ 여전히 7GB 이상이면 오래된 파일 추가 삭제
    while get_directory_size_gb("/persistent") > TRIGGER_GB:
        old_files = get_sorted_old_files(LOG_DIR)
        if not old_files: break
        fpath, _ = old_files[0]
        try:
            os.remove(fpath)
            deleted.append(os.path.basename(fpath))
        except: break

    if deleted:
        with open(DELETED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 삭제된 파일 목록:\n")
            for name in deleted:
                f.write(f"  - {name}\n")
        print(f"[🧹 삭제 완료] 총 {len(deleted)}개 파일 삭제")
    else:
        print("[📁 삭제 없음] 최근 로그만 존재")

# ✅ 실행
auto_delete_old_logs()
