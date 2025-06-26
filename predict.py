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


DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))


# ✅ 클래스 → 기대수익률 중앙값 매핑 (21개 class_ranges 기준)
def class_to_expected_return(cls):
    # ✅ 21개 클래스 수익률 중심값 리스트 (복제 클래스 포함 기준)
    centers = [
        -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
        -0.02, 0.0,  # 중립 (9번 = 0.0)
         0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
         0.75, 1.50, 3.50
    ]

    if isinstance(cls, int) and 0 <= cls < len(centers):
        return centers[cls]
    
    # ✅ 예외 상황 로그로 확인하기 위함
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

def predict(symbol, strategy, source="일반"):
    try:
        print(f"[PREDICT] {symbol}-{strategy} 시작")
        sys.stdout.flush()

        window = find_best_window(symbol, strategy)
        if not isinstance(window, int) or window <= 0:
            return [failed_result(symbol, strategy, "unknown", "윈도우 결정 실패", source)]

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < window + 1:
            return [failed_result(symbol, strategy, "unknown", "데이터 부족", source)]

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < window + 1:
            return [failed_result(symbol, strategy, "unknown", "feature 부족", source)]

        if "volatility" in feat.columns and feat["volatility"].iloc[-1] < 0.00001:
            return [failed_result(symbol, strategy, "unknown", "변화량 없음", source)]

        if feat["close"].nunique() < 3:
            return [failed_result(symbol, strategy, "unknown", "가격 변화 부족", source)]

        if "timestamp" not in feat.columns:
            return [failed_result(symbol, strategy, "unknown", "timestamp 없음", source)]

        raw_close = df["close"].iloc[-1]
        raw_feat = feat.dropna().copy()
        features_only = raw_feat.drop(columns=["timestamp"])
        feat_scaled = MinMaxScaler().fit_transform(features_only)

        if feat_scaled.shape[0] < window:
            return [failed_result(symbol, strategy, "unknown", "시퀀스 부족", source)]

        X_input = feat_scaled[-window:]
        if X_input.shape[0] != window:
            return [failed_result(symbol, strategy, "unknown", "시퀀스 길이 오류", source)]

        X = np.expand_dims(X_input, axis=0)
        if len(X.shape) != 3:
            return [failed_result(symbol, strategy, "unknown", "입력 형상 오류", source)]

        predictions = []
        model_files = {
            m["model"]: os.path.join(MODEL_DIR, m["pt_file"])
            for m in get_available_models()
            if m["symbol"] == symbol and m["strategy"] == strategy
        }

        if not model_files:
            return [failed_result(symbol, strategy, "unknown", "모델 없음", source, X_input)]

        for model_type, path in model_files.items():
            try:
                meta_path = path.replace(".pt", ".meta.json")
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if meta.get("model") != model_type or meta.get("input_size") != X.shape[2]:
                    continue

                weight = get_model_weight(model_type, strategy, symbol)
                if weight <= 0.0:
                    predictions.append(failed_result(symbol, strategy, model_type, "모델 가중치 부족", source, X_input))
                    continue

                model = get_model(model_type, X.shape[2], output_size=NUM_CLASSES).to(DEVICE)
                state = torch.load(path, map_location=DEVICE)
                model.load_state_dict(state)
                model.eval()

                with torch.no_grad():
                    logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

                    recent_freq = get_recent_class_frequencies(strategy=strategy)
                    class_counts = meta.get("class_counts", {}) or {}

                    adjusted_probs = adjust_probs_with_diversity(probs, recent_freq, class_counts)

                    top3_idx = adjusted_probs.argsort()[-3:][::-1]
                    final_idx, best_score = top3_idx[0], -1
                    for idx in top3_idx:
                        diversity_bonus = 1.0 - (recent_freq.get(idx, 0) / (sum(recent_freq.values()) + 1e-6))
                        class_weight = 1.0 + (1.0 - class_counts.get(str(idx), 0) / max(class_counts.values()) if class_counts else 0)
                        score = adjusted_probs[idx] * diversity_bonus * class_weight
                        if score > best_score:
                            final_idx = idx
                            best_score = score

                    pred_class = int(final_idx)
                    expected_return = class_to_expected_return(pred_class)
                    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")

                    log_prediction(
                        symbol=symbol, strategy=strategy,
                        direction=f"Class-{pred_class}", entry_price=raw_close,
                        target_price=raw_close * (1 + expected_return),
                        model=model_type, success=True, reason="예측 완료",
                        rate=expected_return, timestamp=t,
                        volatility=True, source=source,
                        predicted_class=pred_class
                    )

                    result = {
                        "symbol": symbol, "strategy": strategy,
                        "model": model_type, "class": pred_class,
                        "expected_return": expected_return,
                        "price": raw_close, "timestamp": t,
                        "success": True, "source": source,
                        "predicted_class": pred_class,
                        "label": pred_class
                    }

                    try:
                        feature_hash = get_feature_hash(X_input)
                        insert_failure_record(result, feature_hash)
                    except:
                        pass

                    predictions.append(result)

                del model

            except Exception as e:
                predictions.append(failed_result(symbol, strategy, model_type, f"예측 예외: {e}", source, X_input))

        if not predictions:
            return [failed_result(symbol, strategy, "unknown", "모든 모델 예측 실패", source, X_input)]

        return predictions

    except Exception as e:
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
                target_price = float(r.get("target_price", 0))
                if entry_price <= 0 or target_price <= 0:
                    raise ValueError
            except:
                r.update({"status": "fail", "reason": "가격 오류", "return": 0.0})
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

            expected_return = (target_price - entry_price) / (entry_price + 1e-6)
            tolerance = 0.10  # 허용 오차 ±10%
            low = expected_return * (1 - tolerance)
            high = expected_return * (1 + tolerance)
            success = low <= gain <= high if pred_class >= 0 else False

            vol = str(r.get("volatility", "")).lower() in ["1", "true"]
            status = "v_success" if vol and success else "v_fail" if not success and vol else "success" if success else "fail"

            r.update({
                "status": status,
                "reason": f"[cls={pred_class}] 기대={expected_return:.3f}, 실현={gain:.3f}, 허용=({low:.3f}~{high:.3f})",
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

