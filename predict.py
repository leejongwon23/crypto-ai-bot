import os, torch, numpy as np, pandas as pd, datetime, pytz, sys
from sklearn.preprocessing import MinMaxScaler
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import log_prediction
from failure_db import insert_failure_record, load_existing_failure_hashes
from logger import get_feature_hash
from collections import Counter
import pandas as pd

def get_recent_class_frequencies(strategy: str, recent_days: int = 3):
    try:
        path = "/persistent/prediction_log.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df[df["strategy"] == strategy]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]
        return Counter(df["predicted_class"].dropna().astype(int))
    except:
        return Counter()


DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
NUM_CLASSES = 14  # ✅ 14개 클래스 기준

# ✅ 클래스 → 기대수익률 중앙값 매핑
def class_to_expected_return(cls):
    centers = [-0.225, -0.125, -0.085, -0.06, -0.04, -0.0225, -0.0125,
                0.0125, 0.0225, 0.04, 0.06, 0.085, 0.125, 0.225]
    return centers[cls] if 0 <= cls < len(centers) else 0.0

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
            volatility=False,
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
        "predicted_class": -1  # ✅ 반드시 포함됨
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
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < window + 1:
            return [failed_result(symbol, strategy, "unknown", "데이터 부족", source)]

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < window + 1:
            return [failed_result(symbol, strategy, "unknown", "feature 부족", source)]

        if "timestamp" not in feat.columns:
            return [failed_result(symbol, strategy, "unknown", "timestamp 없음", source)]

        raw_close = df["close"].iloc[-1]
        raw_feat = feat.dropna().copy()
        timestamps = raw_feat["timestamp"].reset_index(drop=True)
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

        model_files = {
            f.replace(".pt", "").split("_")[-1]: os.path.join(MODEL_DIR, f)
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pt") and f.startswith(symbol) and strategy in f
        }
        if not model_files:
            return [failed_result(symbol, strategy, "unknown", "모델 없음", source, X_input)]

        predictions = []

        for model_type, path in model_files.items():
            try:
                weight = get_model_weight(model_type, strategy, symbol)
                if weight <= 0.0:
                    continue

                model = get_model(model_type, X.shape[2], output_size=NUM_CLASSES)
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()

                with torch.no_grad():
                    logits = model(torch.tensor(X, dtype=torch.float32))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                    recent_freq = get_recent_class_frequencies(strategy)
                    probs[0] = adjust_probs_with_diversity(probs, recent_freq)

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

                    result = {
                        "symbol": symbol, "strategy": strategy,
                        "model": model_type, "class": pred_class,
                        "expected_return": expected_return,
                        "price": raw_close, "timestamp": t,
                        "success": True, "source": source,
                        "predicted_class": pred_class
                    }

                    try:
                        feature_hash = get_feature_hash(X_input)
                        insert_failure_record(result, feature_hash)
                    except:
                        pass

                    predictions.append(result)

            except Exception as e:
                predictions.append(
                    failed_result(symbol, strategy, model_type, f"예측 예외: {e}", source, X_input)
                )

        if not predictions:
            return [failed_result(symbol, strategy, "unknown", "모든 모델 예측 실패", source, X_input)]

        return predictions

    except Exception as e:
        return [failed_result(symbol, strategy, "unknown", f"예외 발생: {e}", source)]


def adjust_probs_with_diversity(probs, recent_freq: Counter, alpha=0.1):
    """
    probs: (1, NUM_CLASSES) softmax 결과
    recent_freq: 최근 예측 클래스 빈도 Counter
    alpha: 조절 강도 (0.1 = ±10% 정도 조정)
    """
    probs = probs.copy()
    if probs.ndim == 2:
        probs = probs[0]
    total = sum(recent_freq.values()) + 1e-6
    weights = np.array([1.0 - alpha * (recent_freq.get(i, 0) / total) for i in range(len(probs))])
    weights = np.clip(weights, 0.7, 1.3)
    adjusted = probs * weights
    return adjusted / adjusted.sum()

