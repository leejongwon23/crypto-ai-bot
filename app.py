from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return "Yopo server is running"

@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/run")
def run_recommend():
    subprocess.Popen(["python", "recommend.py"])
    return "Recommendation started"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
