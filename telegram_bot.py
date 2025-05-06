# telegram_bot.py
import requests
import os

# ✅ 봇 토큰과 채팅 ID는 환경변수에서 불러오기
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# ✅ 텔레그램 메시지 전송 함수
def send_recommendation(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"❌ 텔레그램 전송 실패: {response.text}")
        else:
            print(f"✅ 텔레그램 전송 완료: {message[:30]}...")
    except Exception as e:
        print(f"⚠️ 전송 중 오류 발생: {e}")
