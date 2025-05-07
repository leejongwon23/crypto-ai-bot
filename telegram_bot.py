from telegram import Bot
import os
import time

# 환경변수에서 텔레그램 설정 가져오기
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

bot = Bot(token=BOT_TOKEN)

# ⏳ 쿨타임 설정
last_sent_time = 0
cooldown_seconds = 3600  # 1시간

def send_recommendation(messages: list):
    global last_sent_time
    now = time.time()

    if now - last_sent_time < cooldown_seconds:
        remaining = int(cooldown_seconds - (now - last_sent_time))
        print(f"⏳ 텔레그램 쿨타임 중... {remaining}초 남음")
        return

    try:
        for msg in messages:
            bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")
            time.sleep(1.5)  # 메시지 전송 간 딜레이
        last_sent_time = now
        print("✅ 텔레그램 전송 완료")
    except Exception as e:
        print(f"❌ [텔레그램 오류] {e}")
