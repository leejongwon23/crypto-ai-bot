import os, torch, numpy as np, pandas as pd, datetime, pytz, sys
from sklearn.preprocessing import MinMaxScaler
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import log_prediction
from failure_db import insert_failure_record, load_existing_failure_hashes
from logger import get_feature_hash

DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
NUM_CLASSES = 16

# 클래스 → 수익률 구간 중앙값 추정 (예: 평가/로그용)
def class_to_expected_return(cls):
    bins = [-0.10, -0.07, -0.05, -0.03, -0.015, -0.005,
             0.005, 0.015, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    return bins[cls] if 0 <= cls < len(bins) else 0.0

def failed_result(symbol, strategy, model_type, reason, source="일반"):
    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_prediction(
            symbol=symbol, strategy=strategy,
            direction="예측실패", entry_price=0, target_price=0,
            model=model_type, success=False, reason=reason,
            rate=0.0, timestamp=t, volatility=False, source=source,
            predicted_class=-1
        )
    except: pass
    return {
        "symbol": symbol, "strategy": strategy, "success": False,
        "reason": reason, "model": model_type, "rate": 0.0,
        "class": -1, "timestamp": t, "source": source
    }

def predict(symbol, strategy, source="일반"):
    try:
        print(f"[PREDICT] {symbol}-{strategy} 시작")
        sys.stdout.flush()
        window = find_best_window(symbol, strategy)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < window + 1:
            return [failed_result(symbol, strategy, "unknown", "데이터 부족", source)]

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < window + 1:
            return [failed_result(symbol, strategy, "unknown", "feature 부족", source)]

        if "timestamp" not in feat.columns:
            return [failed_result(symbol, strategy, "unknown", "timestamp 없음", source)]

        raw_close = df["close"].iloc[-1]
        raw_feat = feat.copy()
        timestamps = raw_feat["timestamp"]
        feat_scaled = MinMaxScaler().fit_transform(raw_feat.drop(columns=["timestamp"]))
        feat = pd.DataFrame(feat_scaled, columns=raw_feat.drop(columns=["timestamp"]).columns)
        feat["timestamp"] = timestamps.values

        if feat.shape[0] < window:
            return [failed_result(symbol, strategy, "unknown", "시퀀스 부족", source)]

        X = np.expand_dims(feat.iloc[-window:].drop(columns=["timestamp"]).values, axis=0)
        if X.shape[1] != window or len(X.shape) != 3:
            return [failed_result(symbol, strategy, "unknown", "시퀀스 형상 오류", source)]

        model_files = {
            f.replace(".pt", "").split("_")[-1]: os.path.join(MODEL_DIR, f)
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pt") and f.startswith(symbol) and strategy in f
        }
        if not model_files:
            return [failed_result(symbol, strategy, "unknown", "모델 없음", source)]

        predictions = []
        failure_hashes = load_existing_failure_hashes()

        for model_type, path in model_files.items():
            try:
                weight = get_model_weight(model_type, strategy, symbol)
                if weight <= 0.0:
                    continue

                model = get_model(model_type, X.shape[2])
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()

                with torch.no_grad():
                    logits = model(torch.tensor(X, dtype=torch.float32))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    pred_class = int(np.argmax(probs))
                    expected_return = class_to_expected_return(pred_class)

                    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
                    log_prediction(
                        symbol=symbol, strategy=strategy,
                        direction=f"Class-{pred_class}", entry_price=raw_close,
                        target_price=raw_close * (1 + expected_return),
                        model=model_type, success=True, reason="예측 완료",
                        rate=expected_return, timestamp=t,
                        volatility=False, source=source,
                        predicted_class=pred_class
                    )
                    predictions.append({
                        "symbol": symbol, "strategy": strategy,
                        "model": model_type, "class": pred_class,
                        "expected_return": expected_return,
                        "price": raw_close,
                        "timestamp": t, "success": True,
                        "source": source
                    })
            except Exception as e:
                failed = failed_result(symbol, strategy, model_type, f"예측 예외: {e}", source)
                try:
                    feature_hash = get_feature_hash(X[0])
                    insert_failure_record(failed, feature_hash)
                except: pass
                predictions.append(failed)

        return predictions if predictions else [failed_result(symbol, strategy, "unknown", "모든 모델 예측 실패", source)]

    except Exception as e:
        return [failed_result(symbol, strategy, "unknown", f"예외 발생: {e}", source)]
