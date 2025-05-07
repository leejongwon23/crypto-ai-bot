import telegram
from telegram.ext import Updater, CommandHandler, CallbackContext
from telegram import Update
import os

# 텔레그램 토큰 (환경변수 또는 직접 입력)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "여기에_토큰_입력")

bot = telegram.Bot(token=TOKEN)

# 메시지 전송 함수 (기존)
def send_message(text):
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "여기에_챗아이디_입력")
    bot.send_message(chat_id=chat_id, text=text)

# 명령어: /run
def handle_run(update: Update, context: CallbackContext):
    from recommend import run_recommendation
    run_recommendation()
    update.message.reply_text("📡 추천 분석이 실행되었습니다.")

# 명령어: /status
def handle_status(update: Update, context: CallbackContext):
    update.message.reply_text("✅ 서버는 정상 작동 중입니다.")

# 명령어: /help
def handle_help(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🤖 사용 가능한 명령어:\n"
        "/run - 추천 분석 실행\n"
        "/status - 서버 상태 확인\n"
        "/help - 도움말 보기"
    )

# 수신 봇 실행 함수
def run_bot():
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("run", handle_run))
    dispatcher.add_handler(CommandHandler("status", handle_status))
    dispatcher.add_handler(CommandHandler("help", handle_help))

    updater.start_polling()
    updater.idle()

# 단독 실행 시 (예: python telegram_bot.py)
if __name__ == "__main__":
    run_bot()
