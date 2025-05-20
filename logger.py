import os
import csv
import datetime
import pandas as pd
from data.utils import get_kline_by_strategy

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
LOG_FILE = os.path.join(PERSIST_DIR, "logs", "train_log.csv")
AUDIT_LOG = os.path.join(PERSIST_DIR, "logs", "evaluation_audit.csv")
MESSAGE_LOG = os.path.join(PERSIST_DIR, "logs", "message_log.csv")
FAILURE_LOG = os.path.join(PERSIST_DIR, "logs", "failure_count.csv")
os.makedirs(os.path.join(PERSIST_DIR, "logs"), exist_ok=True)

EVAL_EXPIRY_BUFFER = 12
STOP_LOSS_PCT = 0.02
model_success_tracker = {}

def get_min_gain(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < 20:
        return {"단기": 0.01, "중기": 0.03, "장기": 0.05}.get(strategy, 0.05)
    v = df["close"].pct_change().rolling(20).std().iloc[-1] if not df.empty else 0.01
    return max(round(v * 1.2, 4), {"단기": 0.005, "중기": 0.01, "장기": 0.02}.get(strategy, 0.03))

def update_model_success(symbol, strategy, model, success):
    key = (symbol, strategy, model)
    if key not in model_success_tracker:
        model_success_tracker[key] = {"success": 0, "fail": 0}
    model_success_tracker[key]["success" if success else "fail"] += 1

def get_model_success_rate(symbol, strategy, model, min_total=10):
    r = model_success_tracker.get((symbol, strategy, model), {"success": 0, "fail": 0})
    total = r["success"] + r["fail"]
    return 0.5 if total < min_total else r["success"] / total

def log_audit(symbol, strategy, status, reason):
    now = datetime.datetime.utcnow().isoformat()
    row = {
        "timestamp": now,
        "symbol": str(symbol),
        "strategy": str(strategy),
        "status": str(status),
        "reason": str(reason)
    }
    write_header = not os.path.exists(AUDIT_LOG)
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=row)
            if write_header: writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[오류] log_audit 실패: {e}")

def log_prediction(symbol, strategy, direction=None, entry_price=0, target_price=0,
                   timestamp=None, confidence=0, model="unknown", success=True, reason="", rate=0.0):
    now = timestamp or datetime.datetime.utcnow().isoformat()
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
        "status": "pending",
        "reason": reason or ""
    }
    if not success:
        log_audit(symbol, strategy, "예측실패", reason)
    write_header = not os.path.exists(PREDICTION_LOG)
    try:
        with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=row)
            if write_header: writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[오류] log_prediction 실패: {e}")

def evaluate_predictions(get_price_fn):
    if not os.path.exists(PREDICTION_LOG): return
    try:
        with open(PREDICTION_LOG, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"[경고] 평가 로그 읽기 실패: {e}")
        return

    now, updated_rows = datetime.datetime.utcnow(), []
    for row in rows:
        if row.get("status") != "pending":
            updated_rows.append(row); continue
        try:
            pred_time = datetime.datetime.fromisoformat(row["timestamp"])
            hours_passed = (now - pred_time).total_seconds() / 3600
            strategy, direction = row["strategy"], row["direction"]
            model, entry_price, rate = row.get("model", "unknown"), float(row.get("entry_price", 0)), float(row.get("rate", 0))
            symbol = row["symbol"]
            eval_hours = get_dynamic_eval_wait(strategy)

            if hours_passed > eval_hours + EVAL_EXPIRY_BUFFER:
                row["status"], row["reason"] = "expired", f"평가 유효시간 초과: {hours_passed:.2f}h"
            elif hours_passed < eval_hours:
                row["reason"] = f"{hours_passed:.2f}h < {eval_hours}h"
            else:
                df = get_kline_by_strategy(symbol, strategy)
                if df is None or df.empty or df[df["timestamp"] >= pred_time].empty:
                    row["status"], row["reason"] = "skip_eval", "평가용 데이터 없음 또는 부족"
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    eval_df = df[df["timestamp"] >= pred_time]
                    price = eval_df["high"].max() if direction == "롱" else eval_df["low"].min()
                    gain = (price - entry_price) / entry_price if direction == "롱" else (entry_price - price) / entry_price
                    success = gain >= rate
                    row["status"] = "success" if success else "fail"
                    row["reason"] = f"수익률 도달: {gain:.4f} ≥ 예측 {rate:.4f}" if success else f"미달: {gain:.4f} < 예측 {rate:.4f}"
                    update_model_success(symbol, strategy, model, success)
                    if not success:
                        if not os.path.exists(WRONG_PREDICTIONS):
                            with open(WRONG_PREDICTIONS, "w", newline="", encoding="utf-8-sig") as wf:
                                writer = csv.writer(wf)
                                writer.writerow(["timestamp", "symbol", "strategy", "direction", "entry_price", "target_price", "gain"])
                        with open(WRONG_PREDICTIONS, "a", newline="", encoding="utf-8-sig") as wf:
                            csv.writer(wf).writerow([row["timestamp"], symbol, strategy, direction, entry_price, row["target_price"], gain])
            log_audit(symbol, strategy, row["status"], row["reason"])
        except Exception as e:
            row["status"], row["reason"] = "skip_eval", f"예외 발생: {e}"
            log_audit(row.get("symbol", "?"), row.get("strategy", "?"), "스킵", row["reason"])
        updated_rows.append(row)

    if updated_rows:
        with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=updated_rows[0])
            writer.writeheader()
            writer.writerows(updated_rows)

def get_dynamic_eval_wait(strategy):
    rate = get_actual_success_rate(strategy)
    return {"단기": 2 if rate >= 0.7 else 4 if rate >= 0.4 else 6,
            "중기": 6 if rate >= 0.7 else 12 if rate >= 0.4 else 24,
            "장기": 24 if rate >= 0.7 else 48 if rate >= 0.4 else 72}.get(strategy, 6)

def get_actual_success_rate(strategy=None, threshold=0.7):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["confidence"] >= threshold]
        df = df[df["status"].isin(["success", "fail"])]
        if strategy and strategy != "전체":
            df = df[df["strategy"] == strategy]
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
        counts = {k: len(df[df["status"] == k]) for k in ["success", "fail", "pending", "skipped", "expired", "invalid_model", "skip_eval"] if k in df["status"].unique()}
        summary = [
            f"📊 전체 예측 수: {len(df)}",
            f"✅ 성공: {counts.get('success', 0)}", f"❌ 실패: {counts.get('fail', 0)}",
            f"⏳ 평가 대기중: {counts.get('pending', 0)}", f"⏭️ 스킵: {counts.get('skipped', 0)}",
            f"⌛ 만료: {counts.get('expired', 0)}", f"⚠️ 모델없음: {counts.get('invalid_model', 0)}",
            f"🟡 평가제외: {counts.get('skip_eval', 0)}",
            f"🎯 성공률: {(counts.get('success', 0) / (counts.get('success', 0) + counts.get('fail', 0)) * 100):.2f}%" if (counts.get('success', 0) + counts.get('fail', 0)) > 0 else "🎯 성공률: 0.00%"
        ]
        for strategy in df["strategy"].unique():
            s = df[df["strategy"] == strategy]
            s_succ, s_fail = len(s[s["status"] == "success"]), len(s[s["status"] == "fail"])
            s_rate = (s_succ / (s_succ + s_fail) * 100) if (s_succ + s_fail) > 0 else 0
            summary.append(f"📌 {strategy} 성공률: {s_rate:.2f}%")
        summary.append("")
        for symbol in df["symbol"].unique():
            s = df[df["symbol"] == symbol]
            s_succ, s_fail = len(s[s["status"] == "success"]), len(s[s["status"] == "fail"])
            s_rate = (s_succ / (s_succ + s_fail) * 100) if (s_succ + s_fail) > 0 else 0
            summary.append(f"📍 {symbol} 성공률: {s_rate:.2f}%")
        return "\n".join(summary)
    except Exception as e:
        return f"[오류] 통계 출력 실패: {e}"

def log_training_result(symbol, strategy, model_name, acc, f1, loss):
    row = {
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol, "strategy": strategy, "model": model_name,
        "accuracy": float(acc), "f1_score": float(f1), "loss": float(loss)
    }
    try:
        pd.DataFrame([row]).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False, encoding="utf-8-sig")
        print(f"[LOG] Training result logged for {symbol} - {strategy} - {model_name}")
    except Exception as e:
        print(f"[오류] 학습 로그 저장 실패: {e}")

# ✅ 헬스체크 통합 진단 함수
def get_overall_health():
    summary = {
        "prediction_count": 0,
        "evaluation_complete": 0,
        "evaluation_pending": 0,
        "success_rate": 0.0,
        "message_count": 0,
        "model_count": 0
    }
    try:
        if os.path.exists(PREDICTION_LOG):
            df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
            summary["prediction_count"] = len(df)
            summary["evaluation_complete"] = len(df[df["status"].isin(["success", "fail"])])
            summary["evaluation_pending"] = len(df[df["status"] == "pending"])
            evals = df[df["status"].isin(["success", "fail"])]
            if len(evals) > 0:
                summary["success_rate"] = round(len(evals[evals["status"] == "success"]) / len(evals) * 100, 2)
        if os.path.exists(MESSAGE_LOG):
            df = pd.read_csv(MESSAGE_LOG, encoding="utf-8-sig")
            summary["message_count"] = len(df)
        if os.path.exists(MODEL_DIR):
            models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
            summary["model_count"] = len(models)
    except Exception as e:
        summary["error"] = f"진단 실패: {e}"
    return summary
