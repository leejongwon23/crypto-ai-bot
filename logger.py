import os
import csv
from datetime import datetime

# ✅ 로그 저장 함수
def log_result(symbol, trend, confidence, entry, target, stop):
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{today}.csv")

    header = ["timestamp", "symbol", "trend", "confidence", "entry", "target", "stop"]
    row = [datetime.now().isoformat(), symbol, trend, confidence, entry, target, stop]

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
