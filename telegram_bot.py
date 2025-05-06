import os
from telegram import Bot

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_recommendation(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ 텔레그램 환경변수 누락")
        return
    bot = Bot(token=BOT_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=msg)
