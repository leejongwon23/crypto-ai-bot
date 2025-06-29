import os, torch, numpy as np, pandas as pd, datetime, pytz, sys
from sklearn.preprocessing import MinMaxScaler
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import log_prediction
from failure_db import insert_failure_record, load_existing_failure_hashes
from logger import get_feature_hash
from config import NUM_CLASSES
from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
from logger import get_available_models
import json

DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
def class_to_expected_return(cls):
    # ✅ class_ranges와 1:1 매핑된 21개 클래스 중심값
    centers = [
        -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
        -0.02, 0.0,   # 중립
         0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
         0.75, 1.50, 3.50
    ]

    if isinstance(cls, int) and 0 <= cls < len(centers):
        return centers[cls]

    print(f"[⚠️ 예상 수익률 계산 오류] 잘못된 클래스: {cls}")
    return 0.0


def failed_result(symbol, strategy, model_type="unknown", reason="", source="일반", X_input=None):
    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_prediction(
            symbol=symbol,
            strategy=strategy,
            direction="예측실패",
            entry_price=0,
            target_price=0,
            model=str(model_type or "unknown"),
            success=False,
            reason=reason,
            rate=0.0,
            timestamp=t,
            return_value=0.0,
            volatility=True,
            source=source,
            predicted_class=-1  # ✅ 반드시 포함됨
        )
    except:
        pass

    result = {
        "symbol": symbol,
        "strategy": strategy,
        "success": False,
        "reason": reason,
        "model": str(model_type or "unknown"),
        "rate": 0.0,
        "class": -1,
        "timestamp": t,
        "source": source,
        "predicted_class": -1,  # ✅ 반드시 포함됨
        "label": -1             # ✅ 학습을 위한 실패 클래스 라벨 명시
    }

    if X_input is not None:
        try:
            feature_hash = get_feature_hash(X_input)
            insert_failure_record(result, feature_hash)
        except:
            pass

    return result

def predict(symbol, strategy, source="일반", model_type=None):
    import os, json, torch, numpy as np, pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from data.utils import get_kline_by_strategy, compute_features
    from model.base_model import get_model
    from model_weight_loader import get_model_weight
    from window_optimizer import find_best_window
    from logger import log_prediction, get_feature_hash
    from failure_db import insert_failure_record
    from config import NUM_CLASSES
    from predict_trigger import get_recent_class_frequencies, adjust_probs_with_diversity
    from logger import get_available_models
    DEVICE = torch.device("cpu")
    MODEL_DIR = "/persistent/models"
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

    try:
        window = find_best_window(symbol, strategy)
        if not isinstance(window, int) or window <= 0:
            return [failed_result(symbol, strategy, "unknown", "윈도우 결정 실패", source)]

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < window + 1:
            return [failed_result(symbol, strategy, "unknown", "데이터 부족", source)]

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < window + 1:
            return [failed_result(symbol, strategy, "unknown", "feature 부족", source)]

        features_only = feat.drop(columns=["timestamp"])
        feat_scaled = MinMaxScaler().fit_transform(features_only)
        if feat_scaled.shape[0] < window:
            return [failed_result(symbol, strategy, "unknown", "시퀀스 부족", source)]

        X_input = feat_scaled[-window:]
        X = np.expand_dims(X_input, axis=0)
        input_size = X.shape[2]

        models = get_available_models()
        results = []

        for m in models:
            if m["symbol"] != symbol or m["strategy"] != strategy:
                continue

            mt = m["model"]
            if model_type and mt != model_type:
                continue

            model_path = os.path.join(MODEL_DIR, m["pt_file"])
            meta_path = model_path.replace(".pt", ".meta.json")

            if not os.path.exists(model_path) or not os.path.exists(meta_path):
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("input_size") != input_size:
                continue

            weight = get_model_weight(mt, strategy, symbol)
            if weight <= 0.0:
                continue

            model = get_model(mt, input_size, NUM_CLASSES).to(DEVICE)
            state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state)
            model.eval()

            with torch.no_grad():
                logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            recent_freq = get_recent_class_frequencies(strategy=strategy)
            class_counts = meta.get("class_counts", {}) or {}
            adjusted_probs = adjust_probs_with_diversity(probs, recent_freq, class_counts)

            pred_class = int(adjusted_probs.argmax())
            expected_return = class_to_expected_return(pred_class)

            log_prediction(
                symbol=symbol,
                strategy=strategy,
                direction=f"Class-{pred_class}",
                entry_price=df["close"].iloc[-1],
                target_price=df["close"].iloc[-1] * (1 + expected_return),
                model=mt,
                success=True,
                reason="예측 완료",
                rate=expected_return,
                timestamp=now_kst().strftime("%Y-%m-%d %H:%M:%S"),
                return_value=expected_return,
                volatility=True,
                source=source,
                predicted_class=pred_class
            )

            feature_hash = get_feature_hash(X_input)
            insert_failure_record({
                "symbol": symbol,
                "strategy": strategy,
                "model": mt,
                "class": pred_class,
                "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
            }, feature_hash)

            results.append({
                "symbol": symbol,
                "strategy": strategy,
                "model": mt,
                "class": pred_class,
                "expected_return": expected_return,
                "success": True
            })

        if not results:
            return [failed_result(symbol, strategy, "unknown", "모델 예측 실패", source)]

        return results

    except Exception as e:
        print(f"[predict 예외] {e}")
        return [failed_result(symbol, strategy, "unknown", f"예외 발생: {e}", source)]

# 📄 predict.py 내부에 추가
import csv, datetime, pytz, os
import pandas as pd
from failure_db import ensure_failure_db, insert_failure_record
from logger import update_model_success

def evaluate_predictions(get_price_fn):
    import csv, os, datetime, pytz
    import pandas as pd
    from failure_db import ensure_failure_db
    from logger import update_model_success

    ensure_failure_db()

    PREDICTION_LOG = "/persistent/prediction_log.csv"
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    date_str = now_kst().strftime("%Y-%m-%d")
    EVAL_RESULT = f"/persistent/logs/evaluation_{date_str}.csv"
    WRONG = f"/persistent/logs/wrong_{date_str}.csv"

    # ✅ 클래스별 수익률 구간 정의
    class_ranges = [
    (-0.99, -0.60), (-0.60, -0.30), (-0.30, -0.20), (-0.20, -0.15),
    (-0.15, -0.10), (-0.10, -0.07), (-0.07, -0.05), (-0.05, -0.03),
    (-0.03, -0.01), (-0.01, 0.01),
    ( 0.01, 0.03), ( 0.03, 0.05), ( 0.05, 0.07), ( 0.07, 0.10),
    ( 0.10, 0.15), ( 0.15, 0.20), ( 0.20, 0.30), ( 0.30, 0.60),
    ( 0.60, 1.00), ( 1.00, 2.00), ( 2.00, 5.00)
    ]

    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}
    updated, evaluated = [], []

    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
        if not rows:
            return
    except:
        return

    for r in rows:
        try:
            if r.get("status") not in [None, "", "pending", "v_pending"]:
                updated.append(r)
                continue

            symbol = r["symbol"]
            strategy = r["strategy"]
            model = r.get("model", "unknown")

            try:
                pred_class = int(float(r.get("predicted_class", -1)))
            except:
                pred_class = -1

            try:
                entry_price = float(r.get("entry_price", 0))
            except:
                r.update({"status": "fail", "reason": "entry_price 오류", "return": 0.0})
                updated.append(r)
                continue

            if entry_price <= 0:
                r.update({"status": "fail", "reason": "entry_price 0이하", "return": 0.0})
                updated.append(r)
                continue

            timestamp = pd.to_datetime(r["timestamp"], utc=True).tz_convert("Asia/Seoul")
            deadline = timestamp + pd.Timedelta(hours=eval_horizon_map.get(strategy, 6))
            now = now_kst()
            if now < deadline:
                r.update({"reason": "⏳ 평가 대기 중", "return": 0.0})
                updated.append(r)
                continue

            df = get_price_fn(symbol, strategy)
            if df is None or "timestamp" not in df.columns:
                r.update({"status": "fail", "reason": "가격 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")
            future_df = df[(df["timestamp"] >= timestamp) & (df["timestamp"] <= deadline)]
            if future_df.empty:
                r.update({"status": "fail", "reason": "미래 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            actual_max = future_df["high"].max()
            gain = (actual_max - entry_price) / (entry_price + 1e-6)

            # ✅ 클래스 구간 평가
            if 0 <= pred_class < len(class_ranges):
                cls_min, cls_max = class_ranges[pred_class]
                success = gain >= cls_min
            else:
                success = False

            vol = str(r.get("volatility", "")).lower() in ["1", "true"]
            status = "v_success" if vol and success else \
                     "v_fail" if not success and vol else \
                     "success" if success else "fail"

            r.update({
                "status": status,
                "reason": f"[cls={pred_class}] class_range=({cls_min:.3f}~{cls_max:.3f}), gain={gain:.3f}",
                "return": round(gain, 5)
            })

            update_model_success(symbol, strategy, model, success)
            evaluated.append(r)

        except Exception as e:
            r.update({"status": "fail", "reason": f"예외: {e}", "return": 0.0})
            updated.append(r)

    updated += evaluated

    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=updated[0].keys()).writerows([updated[0]] + updated[1:])

    if evaluated:
        with open(EVAL_RESULT, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=evaluated[0].keys()).writerows([evaluated[0]] + evaluated[1:])

        failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
        if failed:
            with open(WRONG, "a", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=failed[0].keys()).writerows([failed[0]] + failed[1:])

def get_class_distribution(symbol, strategy, model_type):
    import os, json
    meta_path = f"/persistent/models/{symbol}_{strategy}_{model_type}.meta.json"
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("class_counts", {})
    except Exception as e:
        print(f"[⚠️ 클래스 분포 로드 실패] {meta_path} → {e}")
    return {}

