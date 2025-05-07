# scheduler.py (왕1 보완 기능: 1시간마다 자동 학습 + 분석 + 전송)

from apscheduler.schedulers.blocking import BlockingScheduler
from recommend import recommend_strategy
from telegram_bot import send_recommendation
from train_model import train_model  # 왕1에 존재하는 학습 코드

scheduler = BlockingScheduler()

@scheduler.scheduled_job('interval', hours=1)
def scheduled_task():
    print("⏰ [스케줄러] 1시간 주기 실행 시작")

    try:
        # 모델 재학습
        print("🔁 모델 학습 시작")
        train_model()

        # 전략 추천 실행
        print("🔎 전략 분석 시작")
        messages = recommend_strategy()

        # 텔레그램 전송
        print("📤 텔레그램 전송 시작")
        send_recommendation(messages)

    except Exception as e:
        print(f"❌ 스케줄러 오류: {e}")

# 스케줄러 시작
if __name__ == "__main__":
    print("🚀 스케줄러 시작됨 (1시간 주기)")
    scheduler.start()
