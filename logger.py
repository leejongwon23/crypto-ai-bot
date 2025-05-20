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

def log_prediction(symbol, strategy, direction=None, entry_price=None, target_price=None,
                   timestamp=None, confidence=None, model=None, success=True, reason="", rate=0.0):
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
        "rate": rate or 0,
        "status": "pending",
        "reason": reason or ""
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
            eval_hours = get_dynamic_eval_wait(strategy)
            hours_passed = (now - pred_time).total_seconds() / 3600

            if hours_passed > eval_hours + EVAL_EXPIRY_BUFFER:
                row["status"] = "expired"
                row["reason"] = f"평가 유효시간 초과: {hours_passed:.2f}h"
                log_audit(symbol, strategy, "만료", row["reason"])
                updated_rows.append(row)
                continue

            if hours_passed < eval_hours:
                row["reason"] = f"{hours_passed:.2f}h < {eval_hours}h"
                log_audit(symbol, strategy, "대기중", row["reason"])
                updated_rows.append(row)
                continue

            df = get_kline_by_strategy(symbol, strategy)
            if df is None or df.empty:
                row["status"] = "skip_eval"
                row["reason"] = "평가용 데이터 없음"
                log_audit(symbol, strategy, "스킵", row["reason"])
                updated_rows.append(row)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            eval_df = df[df["timestamp"] >= pred_time]

            if eval_df.empty:
                row["status"] = "skip_eval"
                row["reason"] = "평가 구간 데이터 부족"
                log_audit(symbol, strategy, "스킵", row["reason"])
                updated_rows.append(row)
                continue

            if direction == "롱":
                max_price = eval_df["high"].max()
                gain = (max_price - entry_price) / entry_price
                success = gain >= rate
            elif direction == "숏":
                min_price = eval_df["low"].min()
                gain = (entry_price - min_price) / entry_price
                success = gain >= rate
            else:
                row["status"] = "skip_eval"
                row["reason"] = "방향 정보 없음"
                log_audit(symbol, strategy, "스킵", row["reason"])
                updated_rows.append(row)
                continue

            row["status"] = "success" if success else "fail"
            row["reason"] = (
                f"수익률 도달: {gain:.4f} ≥ 예측 {rate:.4f}" if success
                else f"미달: {gain:.4f} < 예측 {rate:.4f}"
            )
            log_audit(symbol, strategy, "성공" if success else "실패", row["reason"])
            update_model_success(symbol, strategy, model, success)

            if not success:
                write_header = not os.path.exists(WRONG_PREDICTIONS)
                with open(WRONG_PREDICTIONS, "a", newline="", encoding="utf-8-sig") as wf:
                    writer = csv.writer(wf)
                    if write_header:
                        writer.writerow(["timestamp", "symbol", "strategy", "direction", "entry_price", "target_price", "gain"])
                    writer.writerow([
                        row["timestamp"], symbol, strategy, direction,
                        entry_price, row["target_price"], gain
                    ])
            updated_rows.append(row)
        except Exception as e:
            row["status"] = "skip_eval"
            row["reason"] = f"예외 발생: {e}"
            log_audit(row.get("symbol", "?"), row.get("strategy", "?"), "스킵", row["reason"])
            updated_rows.append(row)

    if updated_rows:
        with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=updated_rows[0].keys())
            writer.writeheader()
            writer.writerows(updated_rows)

def get_dynamic_eval_wait(strategy):
    rate = get_actual_success_rate(strategy)
    if strategy == "단기":
        return 2 if rate >= 0.7 else 4 if rate >= 0.4 else 6
    elif strategy == "중기":
        return 6 if rate >= 0.7 else 12 if rate >= 0.4 else 24
    elif strategy == "장기":
        return 24 if rate >= 0.7 else 48 if rate >= 0.4 else 72
    return 6

def get_actual_success_rate(strategy=None, threshold=0.7):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["confidence"] >= threshold]
        df = df[df["status"].isin(["success", "fail"])]
        if strategy and strategy != "전체":
            df = df[df["strategy"] == strategy]
        if df.empty:
            return 0.0
        return len(df[df["status"] == "success"]) / len(df)
    except Exception as e:
        print(f"[오류] get_actual_success_rate 실패: {e}")
        return 0.0

def get_strategy_eval_count(strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        df = df[df["status"].isin(["success", "fail"])]
        return len(df)
    except:
        return 0

def get_strategy_fail_rate(symbol, strategy):
    try:
        df = pd.read_csv(PREDICTION_LOG, encoding="utf-8-sig")
        df = df[(df["strategy"] == strategy) & (df["symbol"] == symbol)]
        df = df[df["status"].isin(["success", "fail"])]
        if df.empty: return 0.0
        return len(df[df["status"] == "fail"]) / len(df)
    except:
        return 0.0

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
        expired = len(df[df["status"] == "expired"])
        invalid = len(df[df["status"] == "invalid_model"])
        skipped_eval = len(df[df["status"] == "skip_eval"])

        success_rate = (success / (success + fail)) * 100 if (success + fail) > 0 else 0

        summary = [
            f"📊 전체 예측 수: {total}",
            f"✅ 성공: {success}",
            f"❌ 실패: {fail}",
            f"⏳ 평가 대기중: {pending}",
            f"⏭️ 스킵: {skipped}",
            f"⌛ 만료: {expired}",
            f"⚠️ 모델없음: {invalid}",
            f"🟡 평가제외: {skipped_eval}",
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
