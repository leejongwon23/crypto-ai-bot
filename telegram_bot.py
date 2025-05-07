# telegram_bot.py (쿨타임 제어 포함)

from telegram import Bot
import os
import time

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

bot = Bot(token=BOT_TOKEN)

# ⏳ 쿨타임 설정 (텔레그램 메시지 제한)
last_sent_time = 0
cooldown_seconds = 3600  # 1시간

def send_recommendation(message):
    global last_sent_time
    now = time.time()

    if now - last_sent_time < cooldown_seconds:
        remaining = int(cooldown_seconds - (now - last_sent_time))
        print(f"⏳ 텔레그램 쿨타임 중... {remaining}초 남음")
        return

    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")
        last_sent_time = now
        print("✅ 텔레그램 전송 완료")
    except Exception as e:
        print(f"❌ [텔레그램 오류] {e}")
