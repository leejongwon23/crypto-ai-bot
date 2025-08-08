import os
import shutil
from datetime import datetime, timedelta

ROOT_DIR = "/persistent"
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DELETED_LOG_PATH = os.path.join(LOG_DIR, "deleted_log.txt")

KEEP_DAYS = 1
DISK_LIMIT_GB = 9.8
TRIGGER_GB = 7.0

DELETE_PREFIXES = ["prediction_", "evaluation_", "wrong_", "model_", "ssl_", "meta_", "evo_"]
EXCLUDE_FILES = {
    "prediction_log.csv", "train_log.csv", "evaluation_result.csv",
    "deleted_log.txt", "wrong_predictions.csv", "fine_tune_target.csv"
}

def get_directory_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)

def get_sorted_old_files(paths):
    files = []
    for dir_path in paths:
        if not os.path.exists(dir_path):
            continue
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
                except:
                    continue
    files.sort(key=lambda x: x[1])  # 오래된 순
    return files

def auto_delete_old_logs():
    now = datetime.now()
    cutoff = now - timedelta(days=KEEP_DAYS)
    deleted = []

    current_gb = get_directory_size_gb(ROOT_DIR)
    if current_gb < TRIGGER_GB:
        print(f"[✅ 용량정상] 현재 사용량: {current_gb:.2f}GB → 정리 안함")
        return

    print(f"[⚠️ 용량초과] {current_gb:.2f}GB → 로그/모델 정리 시작")

    # ✅ 오래된 파일 우선 삭제
    for dir_path in [LOG_DIR, MODEL_DIR]:
        if not os.path.exists(dir_path): continue
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath): continue
            if fname in EXCLUDE_FILES: continue
            if not any(fname.startswith(p) for p in DELETE_PREFIXES): continue

            try:
                # 날짜 포맷이 있을 경우
                date_str = fname.split("_")[-1].replace(".csv", "").strip()
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
            except:
                try:
                    file_date = datetime.fromtimestamp(os.path.getmtime(fpath))
                except:
                    continue

            if file_date < cutoff:
                try:
                    os.remove(fpath)
                    deleted.append(fpath)
                except:
                    continue

    # ✅ 여전히 초과 시 → 가장 오래된 파일부터 삭제
    while get_directory_size_gb(ROOT_DIR) > TRIGGER_GB:
        old_files = get_sorted_old_files([LOG_DIR, MODEL_DIR])
        if not old_files:
            break
        fpath, _ = old_files[0]
        try:
            os.remove(fpath)
            deleted.append(fpath)
        except:
            break

    # ✅ 삭제 로그 기록
    if deleted:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(DELETED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 삭제된 파일 목록:\n")
                for path in deleted:
                    f.write(f"  - {path}\n")
            print(f"[🧹 삭제 완료] 총 {len(deleted)}개 파일 삭제")
        except Exception as e:
            print(f"[⚠️ 삭제 로그 기록 실패] → {e}")
            print(f"[🧹 삭제 완료] 총 {len(deleted)}개 파일 삭제 (로그 기록 생략)")
    else:
        print("[📁 삭제 없음] 최근 파일만 존재")

# ✅ 실행
auto_delete_old_logs()

import os

def cleanup_old_models(max_keep=3):
    model_dir = "/persistent/models"
    if not os.path.exists(model_dir):
        return
    files = sorted(
        [os.path.join(model_dir, f) for f in os.listdir(model_dir)],
        key=os.path.getmtime,
        reverse=True
    )
    for old_file in files[max_keep:]:
        os.remove(old_file)
        print(f"[🗑 모델 삭제] {old_file}")
