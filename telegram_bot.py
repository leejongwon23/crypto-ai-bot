import telegram
import time
from recommend import recommend_strategy

# 텔레그램 봇 설정
TELEGRAM_TOKEN = "여기에_본인_봇_토큰"
CHAT_ID = "여기에_본인_채팅_ID"

bot = telegram.Bot(token=TELEGRAM_TOKEN)

# 쿨타임 설정 (1시간)
cooldown = 3600
last_sent_time = 0

def send_recommendations():
    global last_sent_time
    now = time.time()

    if now - last_sent_time < cooldown:
        print("⏳ 쿨타임 미도래, 메시지 전송 생략")
        return

    messages = recommend_strategy()

    for msg in messages:
        try:
            bot.send_message(chat_id=CHAT_ID, text=msg)
            time.sleep(1.5)  # 텔레그램 API 제한 방지
        except Exception as e:
            print(f"❌ 메시지 전송 실패: {e}")

    last_sent_time = now
    print("✅ 모든 메시지 전송 완료")
