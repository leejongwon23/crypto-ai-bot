import os, csv, datetime, pandas as pd, pytz
from data.utils import get_kline_by_strategy

PERSIST_DIR = "/persistent"
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")
AUDIT_LOG = os.path.join(LOG_DIR, "evaluation_audit.csv")
EVAL_EXPIRY_BUFFER, STOP_LOSS_PCT = 12, 0.02
model_success_tracker = {}

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def get_min_gain(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < 20:
        return {"단기": 0.01, "중기": 0.03, "장기": 0.05}.get(strategy, 0.05)
    v = df["close"].pct_change().rolling(20).std().iloc[-1] if not df.empty else 0.01
    return max(round(v * 1.2, 4), {"단기": 0.005, "중기": 0.01, "장기": 0.02}.get(strategy, 0.03))

def update_model_success(symbol, strategy, model, success):
    key = (symbol, strategy, model)
    model_success_tracker.setdefault(key, {"success": 0, "fail": 0})
    model_success_tracker[key]["success" if success else "fail"] += 1

def get_model_success_rate(symbol, strategy, model, min_total=10):
    r = model_success_tracker.get((symbol, strategy, model), {"success": 0, "fail": 0})
    total = r["success"] + r["fail"]
    return 0.5 if total < min_total else r["success"] / total

def log_audit(symbol, strategy, status, reason):
    row = {
        "timestamp": now_kst().isoformat(),
        "symbol": str(symbol),
        "strategy": str(strategy),
        "status": str(status),
        "reason": str(reason)
    }
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row)
            if not os.path.exists(AUDIT_LOG): w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[오류] log_audit 실패: {e}")

def log_prediction(symbol, strategy, direction=None, entry_price=0, target_price=0,
                   timestamp=None, confidence=0, model="unknown", success=True, reason="", rate=0.0):
    now = timestamp or now_kst().isoformat()
    row = {
        "timestamp": now,
        "symbol": str(symbol or "UNKNOWN"),
        "strategy": str(strategy or "UNKNOWN"),
        "direction": direction or "N/A",
        "entry_price": float(entry_price),
        "target_price": float(target_price),
        "confidence": float(confidence),
        "model": model or "unknown",
        "rate": float(rate),
        "status": "pending" if success else "failed",
        "reason": reason or ""
    }
    log_audit(symbol, strategy, "예측성공" if success else "예측실패", reason)
    try:
        with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=row)
            if not os.path.exists(PREDICTION_LOG): w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[오류] log_prediction 실패: {e}")

def log_training_result(symbol, strategy, model_name, acc, f1, loss):
    row = {
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "strategy": strategy, "model": model_name,
        "accuracy": float(acc), "f1_score": float(f1), "loss": float(loss)
    }
    try:
        pd.DataFrame([row]).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False, encoding="utf-8-sig")
        print(f"[LOG] Training result logged for {symbol} - {strategy} - {model_name}")
    except Exception as e:
        print(f"[오류] 학습 로그 저장 실패: {e}")

def evaluate_predictions(get_price_fn):
    if not os.path.exists(PREDICTION_LOG): return
    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
    except Exception as e:
        print(f"[경고] 평가 로그 읽기 실패: {e}")
        return
    now = now_kst()
    updated = []

    for row in rows:
        if row.get("status") not in ["pending", "failed"]:
            updated.append(row); continue
        try:
            pred_time = datetime.datetime.fromisoformat(row["timestamp"]).astimezone(pytz.timezone("Asia/Seoul"))
            hours = (now - pred_time).total_seconds() / 3600
            symbol, strategy, direction = row["symbol"], row["strategy"], row["direction"]
            model, entry, rate = row.get("model", "unknown"), float(row.get("entry_price", 0)), float(row.get("rate", 0))
            if hours > get_dynamic_eval_wait(strategy) + EVAL_EXPIRY_BUFFER:
                row.update({"status": "expired", "reason": f"평가 유효시간 초과: {hours:.2f}h"})
            elif hours < get_dynamic_eval_wait(strategy):
                row["reason"] = f"{hours:.2f}h < {get_dynamic_eval_wait(strategy)}h"
            elif entry == 0 or model == "unknown" or any(k in row["reason"] for k in ["모델 없음", "기준 미달"]):
                row.update({"status": "invalid_model", "reason": "모델 없음 또는 entry_price=0 또는 기준 미달"})
            else:
                df = get_kline_by_strategy(symbol, strategy)
                if df is None or df.empty or df[df["timestamp"] >= pred_time].empty:
                    row.update({"status": "skip_eval", "reason": "평가용 데이터 부족"})
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
                    eval_df = df[df["timestamp"] >= pred_time]
                    price = eval_df["high"].max() if direction == "롱" else eval_df["low"].min()
                    gain = (price - entry) / entry if direction == "롱" else (entry - price) / entry
                    success = gain >= rate
                    row.update({
                        "status": "success" if success else "fail",
                        "reason": f"수익률 도달: {gain:.4f} ≥ 예측 {rate:.4f}" if success else f"미달: {gain:.4f} < 예측 {rate:.4f}"
                    })
                    update_model_success(symbol, strategy, model, success)
                    if not success:
                        if not os.path.exists(WRONG_PREDICTIONS):
                            with open(WRONG_PREDICTIONS, "w", newline="", encoding="utf-8-sig") as wf:
                                csv.writer(wf).writerow(["timestamp", "symbol", "strategy", "direction", "entry_price", "target_price", "gain"])
                        with open(WRONG_PREDICTIONS, "a", newline="", encoding="utf-8-sig") as wf:
                            csv.writer(wf).writerow([row["timestamp"], symbol, strategy, direction, entry, row["target_price"], gain])
            log_audit(symbol, strategy, row["status"], row["reason"])
        except Exception as e:
            row.update({"status": "skip_eval", "reason": f"예외 발생: {e}"})
            log_audit(symbol, strategy, "스킵", row["reason"])
        updated.append(row)

    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=updated[0]).writerows([updated[0]] + updated[1:])

get_dynamic_eval_wait = lambda s: {"단기": 4, "중기": 24, "장기": 168}.get(s, 6)

def get_actual_success_rate(strategy=None, threshold=0.7):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[(df["confidence"] >= threshold) & df["status"].isin(["success", "fail"])]
        if strategy and strategy != "전체": df = df[df["strategy"] == strategy]
        return 0.0 if df.empty else len(df[df["status"] == "success"]) / len(df)
    except: return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        return len(df[(df["strategy"] == strategy) & df["status"].isin(["success", "fail"])])
    except: return 0

def get_strategy_fail_rate(symbol, strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[(df["strategy"] == strategy) & (df["symbol"] == symbol) & df["status"].isin(["success", "fail"])]
        return 0.0 if df.empty else len(df[df["status"] == "fail"]) / len(df)
    except: return 0.0

def print_prediction_stats():
    if not os.path.exists(PREDICTION_LOG): return "예측 기록이 없습니다."
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        statuses = ["success", "fail", "pending", "failed", "skipped", "expired", "invalid_model", "skip_eval"]
        counts = {k: len(df[df["status"] == k]) for k in statuses}
        summary = [
            f"📊 전체 예측 수: {len(df)}",
            f"✅ 성공: {counts['success']}", f"❌ 실패: {counts['fail']}",
            f"⏳ 평가 대기중: {counts['pending']}", f"⏱ 실패예측: {counts['failed']}",
            f"⏭️ 스킵: {counts['skipped']}", f"⌛ 만료: {counts['expired']}",
            f"⚠️ 모델없음: {counts['invalid_model']}", f"🟡 평가제외: {counts['skip_eval']}",
            f"🌟 성공률: {(counts['success'] / (counts['success'] + counts['fail']) * 100):.2f}%" if (counts['success'] + counts['fail']) else "🌟 성공률: 0.00%"
        ]
        for s in df["strategy"].unique():
            d = df[df["strategy"] == s]
            s_s, s_f = len(d[d["status"] == "success"]), len(d[d["status"] == "fail"])
            rate = (s_s / (s_s + s_f) * 100) if (s_s + s_f) else 0
            summary.append(f"📌 {s} 성공률: {rate:.2f}%")
        summary.append("")
        for s in df["symbol"].unique():
            d = df[df["symbol"] == s]
            s_s, s_f = len(d[d["status"] == "success"]), len(d[d["status"] == "fail"])
            rate = (s_s / (s_s + s_f) * 100) if (s_s + s_f) else 0
            summary.append(f"📍 {s} 성공률: {rate:.2f}%")
        return "\n".join(summary)
    except Exception as e:
        return f"[오류] 통계 출력 실패: {e}"
