import os
import shutil
import time
from datetime import datetime, timedelta

# ✅ 설정
LOG_DIR = "/persistent/logs"
DELETED_LOG_PATH = os.path.join(LOG_DIR, "deleted_log.txt")
DISK_LIMIT_GB = 10  # 전체 용량
TRIGGER_GB = 7      # 정리 트리거: 7GB 이상

KEEP_DAYS = 3
DELETE_PREFIXES = ["prediction_", "evaluation_", "wrong_"]
EXCLUDE_FILES = set([
    "prediction_log.csv",
    "train_log.csv",
    "evaluation_result.csv",
    "deleted_log.txt",
    "wrong_predictions.csv",
    "fine_tune_target.csv",
])

def get_directory_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)

def auto_delete_old_logs():
    now = datetime.now()
    cutoff = now - timedelta(days=KEEP_DAYS)
    deleted = []

    # ✅ 현재 디스크 사용량 계산
    current_size_gb = get_directory_size_gb("/persistent")
    if current_size_gb < TRIGGER_GB:
        print(f"[✅ 용량정상] 현재 사용량: {current_size_gb:.2f}GB → 정리 불필요")
        return

    print(f"[⚠️ 디스크 경고] 사용량: {current_size_gb:.2f}GB → 자동 정리 시작")

    if not os.path.exists(LOG_DIR):
        print(f"[❌ 로그 디렉토리 없음] → {LOG_DIR}")
        return

    for fname in os.listdir(LOG_DIR):
        fpath = os.path.join(LOG_DIR, fname)

        if not os.path.isfile(fpath):
            continue
        if fname in EXCLUDE_FILES:
            continue
        if not any(fname.startswith(p) for p in DELETE_PREFIXES):
            continue

        try:
            date_str = fname.split("_")[-1].replace(".csv", "").strip()
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                os.remove(fpath)
                deleted.append(fname)
        except Exception as e:
            print(f"[⚠️ 삭제 실패] {fname} → {e}")

    if deleted:
        with open(DELETED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 삭제된 파일 목록:\n")
            for name in deleted:
                f.write(f"  - {name}\n")
        print(f"[🧹 삭제 완료] {len(deleted)}개 파일 삭제됨.")
    else:
        print("[✅ 삭제 대상 없음] 최근 로그만 존재합니다.")

auto_delete_old_logs()
