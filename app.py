from flask import Flask
from recommend import main  # 여포 1.4 기준 main() 함수 호출

app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong"

@app.route("/run")
def run():
    try:
        main()
        return "Recommendation started"
    except Exception as e:
        print(f"[ERROR] /run 실패: {e}")
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
