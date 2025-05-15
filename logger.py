import os
import csv
import datetime
import pandas as pd

PERSIST_DIR = "/persistent"
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
LOG_FILE = os.path.join(PERSIST_DIR, "logs", "train_log.csv")
os.makedirs(os.path.join(PERSIST_DIR, "logs"), exist_ok=True)

model_success_tracker = {}

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

STRATEGY_EVAL_CONFIG = {
    "단기": {"gain_pct": 0.03, "hours": 4},
    "중기": {"gain_pct": 0.06, "hours": 24},
    "장기": {"gain_pct": 0.10, "hours": 144}
}
STOP_LOSS_PCT = 0.02

def log_prediction(symbol, strategy, direction, entry_price, target_price, timestamp, confidence, model=None):
    row = {
        "timestamp": timestamp,
        "symbol": symbol,
        "strategy": strategy,
        "direction": direction,
        "entry_price": entry_price,
        "target_price": target_price,
        "confidence": confidence,
        "model": model or "unknown",
        "status": "pending"
    }
    file_exists = os.path.isfile(PREDICTION_LOG)
    with open(PREDICTION_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def evaluate_predictions(get_price_fn):
    if not os.path.exists(PREDICTION_LOG):
        return

    with open(PREDICTION_LOG, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    now = datetime.datetime.utcnow()
    updated_rows = []

    for row in rows:
        if row["status"] != "pending":
            updated_rows.append(row)
            continue

        pred_time = datetime.datetime.fromisoformat(row["timestamp"])
        strategy = row["strategy"]
        direction = row["direction"]
        config = STRATEGY_EVAL_CONFIG.get(strategy, {"gain_pct": 0.06, "hours": 6})
        eval_hours = config["hours"]
        min_gain = config["gain_pct"]
        hours_passed = (now - pred_time).total_seconds() / 3600

        if hours_passed < eval_hours:
            updated_rows.append(row)
            continue

        symbol = row["symbol"]
        entry_price = float(row["entry_price"])
        target_price = float(row["target_price"])
        current_price = get_price_fn(symbol)

        if current_price is None:
            updated_rows.append(row)
            continue

        gain = (current_price - entry_price) / entry_price
        success = False

        if direction == "롱":
            success = gain >= min_gain or gain > -STOP_LOSS_PCT
        elif direction == "숏":
            success = -gain >= min_gain or -gain > -STOP_LOSS_PCT

        row["status"] = "success" if success else "fail"
        update_model_success(symbol, strategy, row.get("model", "unknown"), success)

        if not success:
            with open(WRONG_PREDICTIONS, "a", newline="", encoding="utf-8-sig") as wf:
                writer = csv.writer(wf)
                writer.writerow([
                    row["timestamp"], symbol, strategy, direction,
                    entry_price, target_price, current_price, gain
                ])

        updated_rows.append(row)

    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
        writer.writeheader()
        writer.writerows(updated_rows)

def get_actual_success_rate(strategy, threshold=0.7):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        df = df[df["confidence"] >= threshold]
        if len(df) == 0:
            return 1.0
        success_df = df[df["status"] == "success"]
        return len(success_df) / len(df)
    except Exception as e:
        print(f"[경고] 성공률 계산 실패: {e}")
        return 1.0

def print_prediction_stats():
    if not os.path.exists(PREDICTION_LOG):
        return "예측 기록이 없습니다."

    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        total = len(df)
        success = len(df[df["status"] == "success"])
        fail = len(df[df["status"] == "fail"])
        pending = len(df[df["status"] == "pending"])
        success_rate = (success / (success + fail)) * 100 if (success + fail) > 0 else 0

        summary = [
            f"📊 전체 예측 수: {total}",
            f"✅ 성공: {success}",
            f"❌ 실패: {fail}",
            f"⏳ 평가 대기중: {pending}",
            f"🎯 성공률: {success_rate:.2f}%",
        ]

        for strategy in df["strategy"].unique():
            strat_df = df[df["strategy"] == strategy]
            s = len(strat_df[strat_df["status"] == "success"])
            f = len(strat_df[strat_df["status"] == "fail"])
            rate = (s / (s + f)) * 100 if (s + f) > 0 else 0
            summary.append(f"📌 {strategy} 성공률: {rate:.2f}%")

        return "\n".join(summary)

    except Exception as e:
        return f"[오류] 통계 계산 실패: {e}"

def log_training_result(symbol, strategy, model_name, acc, f1, loss):
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "symbol": symbol,
        "strategy": strategy,
        "model": model_name,
        "accuracy": acc,
        "f1_score": f1,
        "loss": loss
    }

    df = pd.DataFrame([log_entry])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

    print(f"[LOG] Training result logged for {symbol} - {strategy} - {model_name}")
