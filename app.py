from flask import Flask
from recommend import recommend_all

app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong"

@app.route("/run")
def run():
    """
    /run 호출 시:
     1) recommend_all() 으로 30개 코인·3개 전략 전체 분석
     2) predict → format_message → send_message 까지 수행
     3) 최종 완료 응답
    """
    try:
        recommend_all()
        return "Recommendation completed"
    except Exception as e:
        # 혹시 내부에서 예외가 올라오면 로그에 찍고 에러 응답
        print(f"[ERROR] /run 실행 중 예외: {e}")
        return f"Error: {e}", 500

if __name__ == "__main__":
    # Render 내부 헬스체크용 포트 그대로 유지
    app.run(host="0.0.0.0", port=10000)
