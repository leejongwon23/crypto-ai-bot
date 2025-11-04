import os
import requests
import datetime
import pytz
import csv

# =========================
# 안전한 퍼시스트 경로 설정
# =========================
# Render 같은 곳은 /persistent 에 권한이 없을 수 있으므로
# 1) PERSIST_DIR 환경변수 우선
# 2) 없으면 /tmp/persistent 로 폴백
BASE_PERSIST_DIR = os.getenv("PERSIST_DIR", "/tmp/persistent")
LOG_DIR = os.path.join(BASE_PERSIST_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "message_log.csv")

# 로그 디렉터리 생성 시도
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    # 여기서 실패해도 send_message 자체는 돌아가게 해야 함
    print(f"[telegram_bot] 로그 디렉터리 생성 실패: {e}")

# =========================
# 텔레그램 설정
# =========================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))


def send_message(text: str):
    """
    텔레그램으로 메시지를 보낸다.
    - 토큰/채팅ID 없으면 콘솔 + 로그에만 남김
    - 로그 파일은 위에서 만든 안전 경로(/tmp/persistent/logs/...)에 쓴다
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("[ERROR] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 환경변수 누락")
        log_message("전송 실패", "환경변수 누락")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": text,
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
            log_message("전송 실패", f"응답 오류: {resp_json} | 메시지 내용: {text}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] 메시지 전송 실패: {e}")
        log_message("전송 실패", f"예외: {e} | 메시지 내용: {text}")


def log_message(status: str, content: str):
    """
    메시지 전송 결과를 CSV로 남긴다.
    - 디렉터리/파일 생성 실패해도 전체 앱이 죽지 않게 try/except 로 감싼다.
    """
    timestamp = now_kst().strftime("%Y-%m-%d %H:%M:%S")
    try:
        write_header = not os.path.exists(LOG_FILE)
        # 상위 디렉터리가 혹시 없다면 한 번 더 시도
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "status", "message"])
            writer.writerow([timestamp, status, content])
    except Exception as e:
        # 여기서 예외가 나도 앱 전체가 죽으면 안 되므로 콘솔에만 남김
        print(f"[telegram_bot] 로그 기록 실패: {e} | {timestamp} | {status} | {content}")
