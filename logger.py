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
os.makedirs(os.path.join(PERSIST_DIR, "logs"), exist_ok=True)

STRATEGY_HOURS = {"단기": 4, "중기": 24, "장기": 144}
EVAL_EXPIRY_BUFFER = 12
STOP_LOSS_PCT = 0.02
model_success_tracker = {}

def get_min_gain(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < 20:
        return {"단기": 0.01, "중기": 0.03, "장기": 0.05}.get(strategy, 0.05)
    volatility = df["close"].pct_change().rolling(window=20).std()
    v = volatility.iloc[-1] if not volatility.isna().all() else 0.01
    if strategy == "단기":
        return max(round(v * 1.2, 4), 0.005)
    elif strategy == "중기":
        return max(round(v * 1.2, 4), 0.01)
    elif strategy == "장기":
        return max(round(v * 1.2, 4), 0.02)
    return 0.03

def update_model_success(symbol, strategy, model, success: bool):
    key = (symbol, strategy, model)
    if key not in model_success_tracker:
        model_success_tracker[key] = {"success": 0, "fail": 0}
    if success:
        model_success_tracker[key]["success"] += 1
    else:
        model_success_tracker[key]["fail"] += 1

def get_model_success_rate(symbol, strategy, model, min_total=10):
    key = (symbol, strategy, model)
    record = model_success_tracker.get(key, {"success": 0, "fail": 0})
    total = record["success"] + record["fail"]
    if total < min_total:
        return 0.5
    return record["success"] / total

def log_audit(symbol, strategy, status, reason):
    now = datetime.datetime.utcnow().isoformat()
    row = {
        "timestamp": now,
        "symbol": symbol,
        "strategy": strategy,
        "status": status,
        "reason": reason
    }
    write_header = not os.path.exists(AUDIT_LOG)
    with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def log_prediction(symbol, strategy, direction=None, entry_price=None, target_price=None, timestamp=None, confidence=None, model=None, success=True, reason=""):
    now = timestamp or datetime.datetime.utcnow().isoformat()
    row = {
        "timestamp": now,
        "symbol": symbol,
        "strategy": strategy,
        "direction": direction or "N/A",
        "entry_price": entry_price or 0,
        "target_price": target_price or 0,
        "confidence": confidence or 0,
        "model": model or "unknown",
        "status": "pending" if success else "fail"
    }
    if not success:
        log_audit(symbol, strategy, "예측실패", reason)
    write_header = not os.path.exists(PREDICTION_LOG) or os.path.getsize(PREDICTION_LOG) == 0
    with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def evaluate_predictions(get_price_fn):
    if not os.path.exists(PREDICTION_LOG):
        return
    try:
        with open(PREDICTION_LOG, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"[경고] 평가 로그 읽기 실패: {e}")
        return

    now = datetime.datetime.utcnow()
    updated_rows = []
    for row in rows:
        if row.get("status") != "pending":
            updated_rows.append(row)
            continue

        try:
            pred_time = datetime.datetime.fromisoformat(row["timestamp"])
            strategy = row["strategy"]
            direction = row["direction"]
            model = row.get("model", "unknown")
            entry_price = float(row.get("entry_price", 0))
            rate = float(row.get("rate", 0))
            symbol = row["symbol"]
            eval_hours = STRATEGY_HOURS.get(strategy, 6)
            hours_passed = (now - pred_time).total_seconds() / 3600

            if hours_passed > eval_hours + EVAL_EXPIRY_BUFFER:
                row["status"] = "skipped"
                log_audit(symbol, strategy, "스킵", f"평가 유효시간 초과: {hours_passed:.2f}h")
                updated_rows.append(row)
                continue

            if hours_passed < eval_hours:
                log_audit(symbol, strategy, "대기중", f"{hours_passed:.2f}h < {eval_hours}h")
                updated_rows.append(row)
                continue

            if direction not in ["롱", "숏"] or model == "unknown" or entry_price == 0:
                row["status"] = "fail"
                log_audit(symbol, strategy, "실패", "평가 불가: 데이터 부족")
                updated_rows.append(row)
                continue

            model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model}.pt")
            if not os.path.exists(model_path):
                row["status"] = "invalid_model"
                log_audit(symbol, strategy, "실패", f"모델 파일 없음: {model_path}")
                updated_rows.append(row)
                continue

            current_price = get_price_fn(symbol)
            if current_price is None:
                log_audit(symbol, strategy, "실패", "현재가 조회 실패")
                updated_rows.append(row)
                continue

            gain = (current_price - entry_price) / entry_price
            success = gain >= rate if direction == "롱" else -gain >= rate
            row["status"] = "success" if success else "fail"
            update_model_success(symbol, strategy, model, success)

            if not success:
                log_audit(symbol, strategy, "실패", f"수익률 미달: {gain:.4f} < 예측 {rate:.4f}")
                write_header = not os.path.exists(WRONG_PREDICTIONS)
                with open(WRONG_PREDICTIONS, "a", newline="", encoding="utf-8-sig") as wf:
                    writer = csv.writer(wf)
                    if write_header:
                        writer.writerow(["timestamp", "symbol", "strategy", "direction", "entry_price", "target_price", "current_price", "gain"])
                    writer.writerow([
                        row["timestamp"], symbol, strategy, direction,
                        entry_price, row["target_price"], current_price, gain
                    ])
            else:
                log_audit(symbol, strategy, "성공", f"수익률 달성: {gain:.4f} >= 예측 {rate:.4f}")
        except Exception as e:
            log_audit(row.get("symbol", "?"), row.get("strategy", "?"), "실패", f"예외: {e}")
        updated_rows.append(row)

    if updated_rows:
        with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
            writer.writeheader()
            writer.writerows(updated_rows)

def get_actual_success_rate(strategy, threshold=0.7):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[(df["strategy"] == strategy) & (df["confidence"] >= threshold)]
        if df.empty:
            return 0.0
        evaluated = df[df["status"].isin(["success", "fail"])]
        if evaluated.empty:
            return 0.0
        return len(evaluated[evaluated["status"] == "success"]) / len(evaluated)
    except:
        return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        evaluated = df[df["status"].isin(["success", "fail"])]
        return len(evaluated)
    except:
        return 0

def print_prediction_stats():
    if not os.path.exists(PREDICTION_LOG):
        return "예측 기록이 없습니다."
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        total = len(df)
        success = len(df[df["status"] == "success"])
        fail = len(df[df["status"] == "fail"])
        pending = len(df[df["status"] == "pending"])
        skipped = len(df[df["status"] == "skipped"])
        invalid = len(df[df["status"] == "invalid_model"])
        success_rate = (success / (success + fail)) * 100 if (success + fail) > 0 else 0

        summary = [
            f"📊 전체 예측 수: {total}",
            f"✅ 성공: {success}",
            f"❌ 실패: {fail}",
            f"⏳ 평가 대기중: {pending}",
            f"⏭️ 스킵: {skipped}",
            f"⚠️ 모델없음: {invalid}",
            f"🎯 성공률: {success_rate:.2f}%"
        ]

        for strategy in df["strategy"].unique():
            s_df = df[df["strategy"] == strategy]
            s_succ = len(s_df[s_df["status"] == "success"])
            s_fail = len(s_df[s_df["status"] == "fail"])
            s_rate = (s_succ / (s_succ + s_fail)) * 100 if (s_succ + s_fail) > 0 else 0
            summary.append(f"📌 {strategy} 성공률: {s_rate:.2f}%")
        return "\n".join(summary)
    except Exception as e:
        return f"[오류] 통계 출력 실패: {e}"

def log_training_result(symbol, strategy, model_name, acc, f1, loss):
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp,
        "symbol": symbol,
        "strategy": strategy,
        "model": model_name,
        "accuracy": float(acc),
        "f1_score": float(f1),
        "loss": float(loss)
    }
    df = pd.DataFrame([row])
    try:
        df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[오류] 학습 로그 저장 실패: {e}")
    else:
        print(f"[LOG] Training result logged for {symbol} - {strategy} - {model_name}")
