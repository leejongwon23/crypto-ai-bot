from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/run")
def run_recommend():
    try:
        result = subprocess.run(["python", "recommend.py"], capture_output=True, text=True)
        return f"<pre>{result.stdout}</pre>", 200
    except Exception as e:
        return f"[ERROR] recommend.py 실행 실패: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
