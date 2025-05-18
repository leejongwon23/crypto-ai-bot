from flask import Flask, jsonify, request, send_file
from recommend import main
import train
import os
import threading
import datetime
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import traceback
import sys
from telegram_bot import send_message
import logger
from predict_test import test_all_predictions
from data.utils import get_latest_price
import shutil
import time

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_COUNT_LOG = os.path.join(LOG_DIR, "failure_count.csv")

def start_auto_prediction_loop():
    def loop():
        while True:
            try:
                print(f"[AUTO-PREDICT] {datetime.datetime.now()} - main() 실행")
                sys.stdout.flush()
                main()
            except Exception as e:
                print(f"[AUTO-PREDICT ERROR] {e}")
            time.sleep(3600)  # 1시간마다 실행
    threading.Thread(target=loop, daemon=True).start()

def start_scheduler():
    print(">>> start_scheduler() 호출됨")
    sys.stdout.flush()
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Seoul'))

    def run_evaluation():
        print(f"[평가 시작] {datetime.datetime.now()}")
        sys.stdout.flush()
        try:
            logger.evaluate_predictions(get_latest_price)
            print("[평가 완료]")
        except Exception as e:
            print(f"[평가 오류] {e}")

    def train_short():
        print("[단기 학습 시작]")
        threading.Thread(target=train.train_model_loop, args=("단기",), daemon=True).start()

    def train_mid():
        print("[중기 학습 시작]")
        threading.Thread(target=train.train_model_loop, args=("중기",), daemon=True).start()

    def train_long():
        print("[장기 학습 시작]")
        threading.Thread(target=train.train_model_loop, args=("장기",), daemon=True).start()

    scheduler.add_job(run_evaluation, 'cron', minute=20, id='eval_loop', replace_existing=True)
    scheduler.add_job(train_short, 'cron', hour='0,3,6,9,12,15,18,21', minute=30)
    scheduler.add_job(train_mid, 'cron', hour='1,7,13,19', minute=30)
    scheduler.add_job(train_long, 'cron', hour='2,14', minute=30)
    scheduler.add_job(test_all_predictions, 'cron', minute=10, id='predict_test', replace_existing=True)
    scheduler.start()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료")
sys.stdout.flush()

@app.route("/")
def index():
    return "Yopo server is running"

@app.route("/ping")
def ping():
    return "pong"

@app.route("/run")
def run():
    try:
        print("[RUN] main() 실행 시작")
        sys.stdout.flush()
        main()
        print("[RUN] main() 실행 완료")
        sys.stdout.flush()
        return "Recommendation started"
    except Exception as e:
        print("[ERROR] /run 실패:")
        traceback.print_exc()
        sys.stdout.flush()
        return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    try:
        print("[TRAIN-NOW] 전체 학습 즉시 실행 시작")
        sys.stdout.flush()
        threading.Thread(target=train.train_all_models, daemon=True).start()
        return "✅ 모든 코인 + 전략 학습이 지금 바로 시작됐습니다!"
    except Exception as e:
        return f"학습 시작 실패: {e}", 500

@app.route("/train-log")
def train_log():
    try:
        if not os.path.exists(LOG_FILE):
            return "아직 학습 로그가 없습니다."
        with open(LOG_FILE, "r", encoding="utf-8-sig") as f:
            return "<pre>" + f.read() + "</pre>"
    except Exception as e:
        return f"로그 파일을 읽을 수 없습니다: {e}", 500

@app.route("/models")
def list_model_files():
    try:
        if not os.path.exists(MODEL_DIR):
            return "models 폴더가 존재하지 않습니다."
        files = os.listdir(MODEL_DIR)
        if not files:
            return "models 폴더가 비어 있습니다."
        return "<pre>" + "\n".join(files) + "</pre>"
    except Exception as e:
        return f"모델 파일 확인 중 오류 발생: {e}", 500

@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG):
            return jsonify({"error": "prediction_log.csv not found"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        last_10 = df.tail(10).to_dict(orient='records')
        return jsonify(last_10)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-wrong")
def check_wrong():
    try:
        if not os.path.exists(WRONG_PREDICTIONS) or os.path.getsize(WRONG_PREDICTIONS) == 0:
            return jsonify([])
        df = pd.read_csv(WRONG_PREDICTIONS, encoding="utf-8-sig")
        last_10 = df.tail(10).to_dict(orient='records')
        return jsonify(last_10)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-stats")
def check_stats():
    try:
        result = logger.print_prediction_stats()
        if not isinstance(result, str):
            return f"출력 형식 오류: {result}", 500
        formatted = result.replace("📊", "<b>📊</b>").replace("✅", "<b style='color:green'>✅</b>") \
                          .replace("❌", "<b style='color:red'>❌</b>").replace("⏳", "<b>⏳</b>") \
                          .replace("🎯", "<b>🎯</b>").replace("📌", "<b>📌</b>")
        formatted = formatted.replace("\n", "<br>")
        return f"<div style='font-family:monospace; line-height:1.6;'>{formatted}</div>"
    except Exception as e:
        return f"정확도 통계 출력 실패: {e}", 500

@app.route("/reset-all")
def reset_all():
    secret_key = "3572"
    request_key = request.args.get("key")
    if request_key != secret_key:
        return "❌ 인증 실패: 잘못된 접근", 403

    try:
        for file_path in [PREDICTION_LOG, WRONG_PREDICTIONS, LOG_FILE, AUDIT_LOG, MESSAGE_LOG, FAILURE_COUNT_LOG]:
            if os.path.exists(file_path):
                open(file_path, "w").close()

        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)

        return "✅ 예측 기록, 실패 기록, 학습 로그, 평가 로그, 메시지 로그, 실패횟수, 모델 전부 삭제 완료"
    except Exception as e:
        return f"삭제 실패: {e}", 500

@app.route("/audit-log")
def audit_log():
    try:
        if not os.path.exists(AUDIT_LOG):
            return jsonify({"error": "audit log not found"})
        df = pd.read_csv(AUDIT_LOG, encoding="utf-8-sig")
        last_30 = df.tail(30).to_dict(orient="records")
        return jsonify(last_30)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/audit-log-download")
def audit_log_download():
    try:
        if not os.path.exists(AUDIT_LOG):
            return "평가 로그가 없습니다.", 404
        return send_file(AUDIT_LOG, mimetype="text/csv", as_attachment=True, download_name="evaluation_audit.csv")
    except Exception as e:
        return f"다운로드 실패: {e}", 500

@app.route("/health-check")
def health_check():
    results = []
    summary = []

    try:
        if os.path.exists(PREDICTION_LOG):
            df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
            total = len(df)
            done = len(df[df["status"].isin(["success", "fail"])])
            rate = (done / total * 100) if total > 0 else 0
            results.append(f"✅ 예측 기록 OK ({total}건)")
            summary.append(f"- 최근 예측 {total}건 중 {done}건 평가 완료 (평가율 {rate:.1f}%)")
        else:
            results.append("❌ 예측 기록 없음")
            summary.append("- 예측 기록 없음")
    except Exception as e:
        results.append(f"❌ 예측 로그 확인 실패: {e}")

    try:
        if os.path.exists(WRONG_PREDICTIONS) and os.path.getsize(WRONG_PREDICTIONS) > 0:
            df = pd.read_csv(WRONG_PREDICTIONS, encoding="utf-8-sig")
            results.append(f"✅ 실패 예측 기록 OK ({len(df)}건)")
            summary.append("- 실패 예측 기록 누락 없음")
        else:
            results.append("❌ 실패 예측 기록 없음")
            summary.append("- 실패 예측 기록 없음")
    except Exception as e:
        results.append(f"❌ 실패 로그 확인 실패: {e}")

    try:
        if os.path.exists(AUDIT_LOG):
            df = pd.read_csv(AUDIT_LOG, encoding="utf-8-sig")
            results.append(f"✅ 평가 기록 OK ({len(df)}건)")
        else:
            results.append("❌ 평가 기록 없음")
    except Exception as e:
        results.append(f"❌ 평가 로그 확인 실패: {e}")

    try:
        if os.path.exists(MODEL_DIR):
            models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
            results.append(f"✅ 모델 파일 OK ({len(models)}개)")
            summary.append(f"- 모델 파일 {len(models)}개 정상 저장됨")
        else:
            results.append("❌ 모델 폴더 없음")
            summary.append("- 모델 폴더 없음")
    except Exception as e:
        results.append(f"❌ 모델 확인 실패: {e}")

    try:
        if os.path.exists(MESSAGE_LOG) and os.path.getsize(MESSAGE_LOG) > 0:
            df = pd.read_csv(MESSAGE_LOG, encoding="utf-8-sig")
            results.append(f"✅ 메시지 전송 기록 OK (최근 {len(df)}건)")
            summary.append(f"- 최근 메시지 {len(df)}건 전송 완료")
        else:
            results.append("❌ 메시지 전송 기록 없음")
            summary.append("- 메시지 전송 기록 없음")
    except Exception as e:
        results.append(f"❌ 메시지 로그 확인 실패: {e}")

    try:
        for s in ["단기", "중기", "장기"]:
            r = logger.get_actual_success_rate(s, threshold=0.0)
            summary.append(f"- {s} 전략 성공률: {r*100:.1f}%")
    except:
        summary.append("- 전략별 성공률 확인 실패")

    if all(r.startswith("✅") for r in results):
        summary.append("🟢 YOPO는 현재 정상 운영 중입니다. 신뢰하고 사용하셔도 됩니다.")
    else:
        summary.append("⚠️ YOPO에서 일부 이상이 감지되었습니다. 위 내용을 확인하세요.")

    formatted = "<br>".join(results + [""] + summary)
    return f"<div style='font-family:monospace; line-height:1.6;'>{formatted}</div>"

if __name__ == "__main__":
    print(">>> __main__ 진입, 서버 실행 준비")
    sys.stdout.flush()
    start_scheduler()
    start_auto_prediction_loop()
    send_message("[시스템 테스트] YOPO 서버가 정상적으로 실행되었으며 텔레그램 메시지도 전송됩니다.")
    print("✅ 테스트 메시지 전송 완료")
    sys.stdout.flush()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
