import os
import requests
import datetime
import pytz
import csv

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
LOG_FILE = "/persistent/logs/message_log.csv"
os.makedirs("/persistent/logs", exist_ok=True)

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def send_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN 또는 CHAT_ID 환경변수 누락")
        log_message("전송 실패", "환경변수 누락")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": text
    }

    try:
        res = requests.post(url, data=data, timeout=10)
        res.raise_for_status()
        resp_json = res.json()

        if resp_json.get("ok"):
            print("[TELEGRAM] 메시지 전송 성공")
            log_message("전송 성공", text)
        else:
            print(f"[TELEGRAM ERROR] 응답 오류: {resp_json}")
            log_message("전송 실패", f"응답 오류: {resp_json}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] 메시지 전송 실패: {e}")
        log_message("전송 실패", f"예외: {e}")

def log_message(status, content):
    timestamp = now_kst().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "status", "message"])
        writer.writerow([timestamp, status, content])
