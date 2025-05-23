# YOPO 서버 진입점 - 전체 통합 구조 포함
from flask import Flask, jsonify, request, send_file
from recommend import main
import train, os, threading, datetime, pandas as pd, pytz, traceback, sys, shutil, time, csv
from apscheduler.schedulers.background import BackgroundScheduler
from telegram_bot import send_message
from predict_test import test_all_predictions
from predict_trigger import run as trigger_run
from data.utils import SYMBOLS, get_kline_by_strategy

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(LOG_DIR, "message_log.csv")
FAILURE_COUNT_LOG = os.path.join(LOG_DIR, "failure_count.csv")
os.makedirs(LOG_DIR, exist_ok=True)

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def get_symbols_by_volatility(strategy):
    threshold_map = {"단기": 0.003, "중기": 0.005, "장기": 0.008}
    threshold = threshold_map.get(strategy, 0.003)
    selected = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 20:
                continue
            vol = df["close"].pct_change().rolling(window=20).std().iloc[-1]
            if vol and vol >= threshold:
                selected.append(symbol)
        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 변동성 계산 실패: {e}")
    return selected

def start_scheduler():
    print(">>> start_scheduler() 호출됨"); sys.stdout.flush()
    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))

    학습_스케줄 = [
        (1, 30, "단기"),
        (3, 30, "장기"),
        (6, 0, "중기"),
        (9, 0, "단기"),
        (11, 0, "중기"),
        (13, 0, "장기"),
        (15, 0, "단기"),
        (17, 0, "중기"),
        (19, 0, "장기"),
        (22, 30, "단기"),
    ]
    for h, m, s in 학습_스케줄:
        scheduler.add_job(lambda s=s: threading.Thread(target=train.train_model_loop, args=(s,), daemon=True).start(),
                          'cron', hour=h, minute=m)

    예측_스케줄 = [
        (7, 30, "단기"), (7, 30, "중기"), (7, 30, "장기"),
        (10, 30, "단기"), (10, 30, "중기"),
        (12, 30, "중기"),
        (14, 30, "장기"),
        (16, 30, "단기"),
        (18, 30, "중기"),
        (21, 0, "단기"), (21, 0, "중기"), (21, 0, "장기"),
        (0, 0, "단기"), (0, 0, "중기"),
    ]
    for h, m, s in 예측_스케줄:
        scheduler.add_job(lambda s=s: threading.Thread(target=main, args=(s,), daemon=True).start(),
                          'cron', hour=h, minute=m)

    scheduler.add_job(lambda: __import__('logger').evaluate_predictions(None), 'cron', minute=20)
    scheduler.add_job(test_all_predictions, 'cron', minute=10)
    scheduler.add_job(trigger_run, 'interval', minutes=30)
    scheduler.start()

app = Flask(__name__)
print(">>> Flask 앱 생성 완료"); sys.stdout.flush()

@app.route("/")
def index():
    return "Yopo server is running"

@app.route("/ping")
def ping():
    return "pong"

@app.route("/run")
def run():
    try:
        print("[RUN] main() 실행 시작"); sys.stdout.flush()
        main()
        print("[RUN] main() 실행 완료"); sys.stdout.flush()
        return "Recommendation started"
    except Exception as e:
        print("[ERROR] /run 실패:"); traceback.print_exc(); sys.stdout.flush()
        return f"Error: {e}", 500

@app.route("/train-now")
def train_now():
    try:
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
        return "<pre>" + "\n".join(files) + "</pre>" if files else "models 폴더가 비어 있습니다."
    except Exception as e:
        return f"모델 파일 확인 중 오류 발생: {e}", 500

@app.route("/check-log")
def check_log():
    try:
        if not os.path.exists(PREDICTION_LOG):
            return jsonify({"error": "prediction_log.csv not found"})
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-wrong")
def check_wrong():
    try:
        if not os.path.exists(WRONG_PREDICTIONS):
            return jsonify([])
        df = pd.read_csv(WRONG_PREDICTIONS, encoding="utf-8-sig")
        return jsonify(df.tail(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/check-stats")
def check_stats():
    try:
        result = __import__('logger').print_prediction_stats()
        if not isinstance(result, str):
            return f"출력 형식 오류: {result}", 500
        for s, r in {"📊": "<b>📊</b>", "✅": "<b style='color:green'>✅</b>",
                     "❌": "<b style='color:red'>❌</b>", "⏳": "<b>⏳</b>",
                     "🎯": "<b>🎯</b>", "📌": "<b>📌</b>"}.items():
            result = result.replace(s, r)
        return f"<div style='font-family:monospace; line-height:1.6;'>" + result.replace(chr(10), "<br>") + "</div>"
    except Exception as e:
        return f"정확도 통계 출력 실패: {e}", 500

@app.route("/reset-all")
def reset_all():
    if request.args.get("key") != "3572":
        return "❌ 인증 실패: 잘못된 접근", 403
    try:
        def clear(path, headers):
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=headers).writeheader()
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)
        clear(PREDICTION_LOG, ["symbol", "strategy", "direction", "price", "target", "timestamp", "confidence", "model", "success", "reason", "status"])
        clear(WRONG_PREDICTIONS, ["symbol", "strategy", "reason", "timestamp"])
        clear(LOG_FILE, ["timestamp", "symbol", "strategy", "model", "accuracy", "f1", "loss"])
        clear(AUDIT_LOG, ["timestamp", "symbol", "strategy", "result", "status"])
        clear(MESSAGE_LOG, ["timestamp", "symbol", "strategy", "message"])
        clear(FAILURE_COUNT_LOG, ["symbol", "strategy", "failures"])
        return "✅ 초기화 완료 (헤더 포함)"
    except Exception as e:
        return f"삭제 실패: {e}", 500

@app.route("/audit-log")
def audit_log():
    try:
        if not os.path.exists(AUDIT_LOG):
            return jsonify({"error": "audit log not found"})
        df = pd.read_csv(AUDIT_LOG, encoding="utf-8-sig")
        return jsonify(df.tail(30).to_dict(orient="records"))
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

@app.route("/yopo-health")
def yopo_health():
    import pandas as pd
    import os, datetime, pytz
    from collections import defaultdict

    def now_kst():
        return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

    def format_percent(val):
        return f"{val:.1f}%" if pd.notna(val) else "0.0%"

    strategies = ["단기", "중기", "장기"]
    logs = {}
    for name, path in {
        "pred": PREDICTION_LOG,
        "train": LOG_FILE,
        "audit": AUDIT_LOG,
        "msg": MESSAGE_LOG,
    }.items():
        logs[name] = pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else pd.DataFrame()

    strategy_html_blocks = []
    abnormal_msgs = []

    for strategy in strategies:
        pred = logs["pred"][logs["pred"]["strategy"] == strategy] if not logs["pred"].empty else pd.DataFrame()
        train = logs["train"][logs["train"]["strategy"] == strategy] if not logs["train"].empty else pd.DataFrame()
        audit = logs["audit"][logs["audit"]["strategy"] == strategy] if not logs["audit"].empty else pd.DataFrame()

        # 모델 수
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt") and strategy in f]
        model_count = len(models)

        # 최근 시각
        recent_train = train["timestamp"].iloc[-1] if not train.empty else "없음"
        recent_pred = pred["timestamp"].iloc[-1] if not pred.empty else "없음"
        recent_eval = audit[audit["strategy"] == strategy]["timestamp"].iloc[-1] if not audit.empty else "없음"

        # 상태 수
        pred_success = len(pred[pred["status"] == "success"])
        pred_fail = len(pred[pred["status"] == "fail"])
        pred_pending = len(pred[pred["status"] == "pending"])
        pred_failed = len(pred[pred["status"] == "failed"])
        total_preds = pred_success + pred_fail + pred_pending + pred_failed

        # 작동 여부 판단
        predict_ok = "✅" if total_preds > 0 else "❌"
        eval_ok = "✅" if pred_success + pred_fail > 0 else "⏳"
        train_ok = "✅" if recent_train != "없음" else "❌"

        # 일반/변동성 분리
        is_vol = pred["symbol"].str.contains("_v", na=False)
        pred_nvol = pred[~is_vol]
        pred_vol = pred[is_vol]

        def get_perf(df):
            succ = len(df[df["status"] == "success"])
            fail = len(df[df["status"] == "fail"])
            r_avg = df["return"].mean() if "return" in df.columns and not df.empty else 0.0
            total = succ + fail
            return {
                "succ": succ,
                "fail": fail,
                "succ_rate": succ / total * 100 if total else 0,
                "fail_rate": fail / total * 100 if total else 0,
                "r_avg": r_avg,
            }

        perf_nvol = get_perf(pred_nvol)
        perf_vol = get_perf(pred_vol)
                # 이상 감지
        if perf_nvol["fail_rate"] > 50:
            abnormal_msgs.append(f"⚠️ {strategy} 일반 예측 실패율 {perf_nvol['fail_rate']:.1f}%")
        if perf_vol["fail_rate"] > 50:
            abnormal_msgs.append(f"⚠️ {strategy} 변동성 예측 실패율 {perf_vol['fail_rate']:.1f}%")
        if eval_ok != "✅":
            abnormal_msgs.append(f"❌ {strategy} 평가 작동 안됨")

        block = f"""
        <div style='border:1px solid #aaa; margin:12px; padding:10px; font-family:monospace;'>
        <b>📌 전략: {strategy}</b><br>
        - 모델 수: {model_count}<br>
        - 최근 학습: {recent_train}<br>
        - 최근 예측: {recent_pred}<br>
        - 최근 평가: {recent_eval}<br>
        - 예측 수: {total_preds} (✅ {pred_success} / ❌ {pred_fail} / ⏳ {pred_pending} / 🛑 {pred_failed})<br>
        <br><b>🎯 일반 예측 성능</b><br>
        - 성공률: {format_percent(perf_nvol['succ_rate'])} / 실패율: {format_percent(perf_nvol['fail_rate'])} / 수익률: {perf_nvol['r_avg']:.2f}%<br>
        <b>🌪️ 변동성 예측 성능</b><br>
        - 성공률: {format_percent(perf_vol['succ_rate'])} / 실패율: {format_percent(perf_vol['fail_rate'])} / 수익률: {perf_vol['r_avg']:.2f}%<br>
        <br>
        - 예측 작동: {predict_ok} / 평가 작동: {eval_ok} / 학습 작동: {train_ok}<br>
        </div>
        """
        strategy_html_blocks.append(block)

        # 최근 예측 10건 테이블
        recent10 = pred.tail(10)[["timestamp", "symbol", "direction", "return", "confidence", "status"]]
        rows = []
        for _, row in recent10.iterrows():
            status_icon = {"success": "✅", "fail": "❌", "pending": "⏳", "failed": "🛑"}.get(row["status"], "")
            rows.append(f"<tr><td>{row['timestamp']}</td><td>{row['symbol']}</td><td>{row['direction']}</td><td>{row['return']:.2f}%</td><td>{row['confidence']}%</td><td>{status_icon}</td></tr>")
        table = "<table border='1' style='font-family:monospace; margin-bottom:20px;'><tr><th>시각</th><th>종목</th><th>방향</th><th>수익률</th><th>신뢰도</th><th>상태</th></tr>" + "".join(rows) + "</table>"
        strategy_html_blocks.append(f"<b>📋 {strategy} 최근 예측 10건</b><br>{table}")

    # 종합 진단 요약
    overall = "🟢 전체 정상 작동 중" if not abnormal_msgs else "🔴 진단 요약:<br>" + "<br>".join(abnormal_msgs)

    return f"<div style='font-family:monospace; line-height:1.6;'><b>{overall}</b><hr>" + "".join(strategy_html_blocks) + "</div>"

        


if __name__ == "__main__":
    print(">>> __main__ 진입, 서버 실행 준비"); sys.stdout.flush()
    threading.Thread(target=start_scheduler, daemon=True).start()
    threading.Thread(target=lambda: send_message("[시스템 시작] YOPO 서버가 정상적으로 실행되었습니다."), daemon=True).start()
    print("✅ 서버 초기화 완료 (정기 예측 루프 포함)"); sys.stdout.flush()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
