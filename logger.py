import os
import csv
import datetime

# ✅ Persistent 경로로 변경
PERSIST_DIR = "/persistent"
PREDICTION_LOG = os.path.join(PERSIST_DIR, "prediction_log.csv")
WRONG_PREDICTIONS = os.path.join(PERSIST_DIR, "wrong_predictions.csv")
THRESHOLD_TOLERANCE = 0.01  # 예: 목표 수익률의 99% 이상 도달 시 성공 처리

# ✅ 전략별 평가 대기 시간 설정 (단기: 3h, 중기: 6h, 장기: 12h)
STRATEGY_LIMIT_HOURS = {
    "단기": 3,
    "중기": 6,
    "장기": 12
}

def log_prediction(symbol, strategy, direction, entry_price, target_price, timestamp, confidence):
    row = {
        "timestamp": timestamp,
        "symbol": symbol,
        "strategy": strategy,
        "direction": direction,
        "entry_price": entry_price,
        "target_price": target_price,
        "confidence": confidence,
        "status": "pending"
    }

    os.makedirs(PERSIST_DIR, exist_ok=True)
    file_exists = os.path.isfile(PREDICTION_LOG)
    with open(PREDICTION_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def evaluate_predictions(get_price_fn):
    if not os.path.exists(PREDICTION_LOG):
        return

    with open(PREDICTION_LOG, "r") as f:
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
        limit_hours = STRATEGY_LIMIT_HOURS.get(strategy, 6)
        hours_passed = (now - pred_time).total_seconds() / 3600
        if hours_passed < limit_hours:
            updated_rows.append(row)
            continue

        symbol = row["symbol"]
        entry_price = float(row["entry_price"])
        target_price = float(row["target_price"])
        direction = row["direction"]

        current_price = get_price_fn(symbol)
        if current_price is None:
            updated_rows.append(row)
            continue

        actual_gain = (current_price - entry_price) / entry_price
        expected_gain = (target_price - entry_price) / entry_price
        if direction == "숏":
            actual_gain *= -1
            expected_gain *= -1

        success = actual_gain >= expected_gain * (1 - THRESHOLD_TOLERANCE)
        row["status"] = "success" if success else "fail"

        if not success:
            with open(WRONG_PREDICTIONS, "a", newline="") as wf:
                writer = csv.writer(wf)
                writer.writerow([
                    row["timestamp"], symbol, row["strategy"], direction,
                    entry_price, target_price, current_price, actual_gain
                ])

        updated_rows.append(row)

    with open(PREDICTION_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
        writer.writeheader()
        writer.writerows(updated_rows)

def get_actual_success_rate(strategy, threshold=0.7):
    import pandas as pd
    try:
        df = pd.read_csv(PREDICTION_LOG)
        df = df[df["strategy"] == strategy]
        df = df[df["confidence"] >= threshold]

        if len(df) == 0:
            return 1.0  # 데이터 부족 시 기본값

        success_df = df[df["status"] == "success"]
        return len(success_df) / len(df)
    except Exception as e:
        print(f"[경고] 성공률 계산 실패: {e}")
        return 1.0
