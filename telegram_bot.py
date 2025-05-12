import os
import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_message(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN 또는 CHAT_ID 환경변수 누락")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": text
    }

    try:
        res = requests.post(url, data=data, timeout=10)
        res.raise_for_status()
        print("[TELEGRAM] 메시지 전송 성공")
        print(f"[텔레그램 응답] {res.status_code} - {res.text}")  # ✅ 로그 추가
    except Exception as e:
        print(f"[TELEGRAM ERROR] 메시지 전송 실패: {e}")
