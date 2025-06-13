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
NUM_CLASSES = 21  # 🔄 반드시 전체 구조와 통일

# ✅ 클래스 → 기대수익률 중앙값 매핑 (21개 class_ranges 기준)
def class_to_expected_return(cls):
    centers = [
        -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
        -0.02, 0.0,  # 중립
         0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
         0.75, 1.50, 3.50
    ]
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
    import torch
    import numpy as np
    import pandas as pd
    import os, sys
    from sklearn.preprocessing import MinMaxScaler
    from logger import log_prediction, get_recent_class_frequencies
    from failure_db import insert_failure_record
    from model_weight_loader import get_model_weight
    from model.base_model import get_model
    from window_optimizer import find_best_window
    from data.utils import get_kline_by_strategy, compute_features
    from logger import get_feature_hash
    from predict_test import adjust_probs_with_diversity
    from datetime import datetime
    import pytz

    # ✅ 통일된 설정
    DEVICE = torch.device("cpu")
    MODEL_DIR = "/persistent/models"
    NUM_CLASSES = 21
    now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))

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

                    # ✅ 예외 방지: class_to_expected_return 통일 import 필요
                    from model.base_model import class_to_expected_return
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
NUM_CLASSES = 21  # 🔄 반드시 전체 구조와 통일

# ✅ 클래스 → 기대수익률 중앙값 매핑 (21개 class_ranges 기준)
def class_to_expected_return(cls):
    centers = [
        -0.80, -0.45, -0.25, -0.175, -0.125, -0.085, -0.06, -0.04,
        -0.02, 0.0,  # 중립
         0.02, 0.04, 0.06, 0.085, 0.125, 0.175, 0.25, 0.40,
         0.75, 1.50, 3.50
    ]
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
    import torch
    import numpy as np
    import pandas as pd
    import os, sys
    from sklearn.preprocessing import MinMaxScaler
    from logger import log_prediction, get_recent_class_frequencies
    from failure_db import insert_failure_record
    from model_weight_loader import get_model_weight
    from model.base_model import get_model
    from window_optimizer import find_best_window
    from data.utils import get_kline_by_strategy, compute_features
    from logger import get_feature_hash
    from predict_test import adjust_probs_with_diversity
    from datetime import datetime
    import pytz

    # ✅ 통일된 설정
    DEVICE = torch.device("cpu")
    MODEL_DIR = "/persistent/models"
    NUM_CLASSES = 21
    now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))

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

                    # ✅ 예외 방지: class_to_expected_return 통일 import 필요
                    from model.base_model import class_to_expected_return
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
# 📄 predict.py 내부에 추가
import csv, datetime, pytz, os
import pandas as pd
from failure_db import ensure_failure_db, insert_failure_record
from logger import update_model_success

def evaluate_predictions(get_price_fn):
    ensure_failure_db()

    PREDICTION_LOG = "/persistent/prediction_log.csv"
    EVAL_RESULT = "/persistent/evaluation_result.csv"
    WRONG = "/persistent/wrong_predictions.csv"
    NUM_CLASSES = 21
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

    try:
        rows = list(csv.DictReader(open(PREDICTION_LOG, "r", encoding="utf-8-sig")))
        if not rows:
            print("[스킵] 예측 결과 없음 → 평가 건너뜀")
            return
    except Exception as e:
        print(f"[예측 평가 로드 오류] {e}")
        return

    updated, evaluated = [], []
    eval_horizon_map = {"단기": 4, "중기": 24, "장기": 168}
    class_ranges = [
        (-1.00, -0.60), (-0.60, -0.30), (-0.30, -0.20), (-0.20, -0.15),
        (-0.15, -0.10), (-0.10, -0.07), (-0.07, -0.05), (-0.05, -0.03),
        (-0.03, -0.01), (-0.01,  0.01), ( 0.01,  0.03), ( 0.03,  0.05),
        ( 0.05,  0.07), ( 0.07,  0.10), ( 0.10,  0.15), ( 0.15,  0.20),
        ( 0.20,  0.30), ( 0.30,  0.50), ( 0.50,  1.00), ( 1.00,  2.00),
        ( 2.00,  5.00)
    ]
    assert len(class_ranges) == NUM_CLASSES

    for r in rows:
        try:
            if r.get("status") not in ["pending", "failed", "v_pending", "v_failed"]:
                updated.append(r)
                continue

            symbol = r.get("symbol")
            strategy = r.get("strategy")
            model = r.get("model", "unknown")
            entry_price = float(r.get("entry_price", 0))

            try:
                raw_val = r.get("predicted_class", "")
                if raw_val == "" or str(raw_val).lower() in ["nan", "none"]:
                    raise ValueError("빈 predicted_class")
                pred_class = int(float(raw_val))
            except:
                pred_class = -1

            if pred_class == -1:
                r.update({
                    "status": "skip_eval",
                    "reason": "유효하지 않은 클래스",
                    "return": 0.0
                })
                updated.append(r)
                continue

            timestamp = pd.to_datetime(r["timestamp"], utc=True).tz_convert("Asia/Seoul")
            horizon = eval_horizon_map.get(strategy, 6)
            deadline = timestamp + pd.Timedelta(hours=horizon)
            now = now_kst()

            df = get_price_fn(symbol, strategy)
            if df is None or df.empty or "timestamp" not in df.columns:
                r.update({"status": "skip_eval", "reason": "가격 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Seoul")

            if now < deadline:
                r.update({"reason": f"⏳ 평가 대기 중 ({now.strftime('%H:%M')} < {deadline.strftime('%H:%M')})", "return": 0.0})
                updated.append(r)
                continue

            future_df = df[(df["timestamp"] >= timestamp) & (df["timestamp"] <= deadline)]
            if future_df.empty:
                r.update({"status": "skip_eval", "reason": "미래 구간 데이터 없음", "return": 0.0})
                updated.append(r)
                continue

            actual_max = future_df["high"].max()
            gain = (actual_max - entry_price) / (entry_price + 1e-6)

            if 0 <= pred_class < NUM_CLASSES:
                low, high = class_ranges[pred_class]
                success = gain >= low and abs(gain) >= 0.01
            else:
                success = False
                low, high = 0.0, 0.0

            vol_raw = str(r.get("volatility", "")).strip().lower()
            vol = vol_raw in ["1", "true", "yes"]
            status = "v_success" if vol and success else "v_fail" if vol else "success" if success else "fail"

            r.update({
                "status": status,
                "reason": f"예측={pred_class} / 구간=({low:.3f}~{high:.3f}) / 실현={gain:.4f}",
                "return": round(gain, 5)
            })
            update_model_success(symbol, strategy, model, success)
            evaluated.append(dict(r))

        except Exception as e:
            r.update({"status": "skip_eval", "reason": f"예외 발생: {e}", "return": 0.0})
            updated.append(r)

    if evaluated:
        with open(EVAL_RESULT, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=evaluated[0].keys())
            if os.stat(EVAL_RESULT).st_size == 0:
                w.writeheader()
            w.writerows(evaluated)

        failed = [r for r in evaluated if r["status"] in ["fail", "v_fail"]]
        if failed:
            with open(WRONG, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=failed[0].keys())
                if os.stat(WRONG).st_size == 0:
                    w.writeheader()
                w.writerows(failed)

    with open(PREDICTION_LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=updated[0].keys())
        w.writeheader()
        w.writerows(updated)

