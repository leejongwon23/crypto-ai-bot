import os
import torch
import numpy as np
import datetime
import pytz
from sklearn.metrics import log_loss

from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import get_min_gain

DEVICE = torch.device("cpu")
STOP_LOSS_PCT = 0.02
MODEL_DIR = "/persistent/models"

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def failed_result(symbol, strategy, reason):
    return {
        "symbol": symbol, "strategy": strategy, "success": False,
        "reason": reason, "direction": "롱", "model": "ensemble",
        "confidence": 0.0, "rate": 0.0, "price": 1.0,
        "target": 1.0, "stop": 1.0,
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
    }

def predict(symbol, strategy):
    try:
        best_window = find_best_window(symbol, strategy)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < best_window + 1:
            return failed_result(symbol, strategy, "데이터 부족")

        features = compute_features(df)
        if features is None or len(features) < best_window + 1:
            return failed_result(symbol, strategy, "feature 부족")

        X_raw = features.iloc[-best_window:].values
        if X_raw.shape[0] != best_window:
            return failed_result(symbol, strategy, "시퀀스 길이 오류")
        X = np.expand_dims(X_raw, axis=0)
        if len(X.shape) != 3:
            return failed_result(symbol, strategy, "시퀀스 형상 오류")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        input_size = X.shape[2]

        model_paths = {
            file.replace(f"{symbol}_{strategy}_", "").replace(".pt", ""): os.path.join(MODEL_DIR, file)
            for file in os.listdir(MODEL_DIR)
            if file.endswith(".pt") and file.startswith(f"{symbol}_{strategy}_")
        }
        if not model_paths:
            return failed_result(symbol, strategy, "모델 없음")

        min_gain = get_min_gain(symbol, strategy)
        results = []

        for model_type, model_path in model_paths.items():
            try:
                model = get_model(model_type=model_type, input_size=input_size)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    signal, confidence = model(X_tensor)
                    if signal is None or confidence is None:
                        continue
                    signal = float(signal.squeeze().item())
                    confidence = float(confidence.squeeze().item())

                    if not (0 <= signal <= 1):
                        continue
                    # 중립 구간: 예측 무의미한 영역
                    if 0.48 <= signal <= 0.52:
                        continue

                    direction = "롱" if signal > 0.5 else "숏"
                    raw_rate = abs(signal - 0.5) * 2
                    weight = get_model_weight(model_type, strategy)

                    try:
                        loss_penalty = log_loss([1 if signal > 0.5 else 0], [np.clip(signal, 1e-6, 1 - 1e-6)], labels=[0, 1])
                        confidence_penalty = max(0.1, 1.0 - loss_penalty)
                    except Exception as e:
                        confidence_penalty = confidence  # fallback
                        print(f"[log_loss 예외] {symbol}-{strategy}-{model_type}: {e}")

                    final_conf = (confidence + confidence_penalty) / 2
                    rate = raw_rate * min_gain * final_conf
                    score = final_conf * weight * rate

                    results.append({
                        "model": model_type,
                        "direction": direction,
                        "confidence": final_conf,
                        "weight": weight,
                        "score": score,
                        "rate": rate
                    })
            except Exception as e:
                print(f"[모델 예측 실패] {symbol}-{strategy}-{model_type}: {e}")
                continue

        if not results:
            return failed_result(symbol, strategy, "모든 모델 예측 실패")

        dir_count = {"롱": 0, "숏": 0}
        for r in results:
            dir_count[r["direction"]] += 1

        if dir_count["롱"] >= 2:
            final_direction = "롱"
        elif dir_count["숏"] >= 2:
            final_direction = "숏"
        elif len(results) == 1:
            final_direction = results[0]["direction"]
        else:
            return failed_result(symbol, strategy, "모델 방향 불일치")

        valid = [r for r in results if r["direction"] == final_direction]
        avg_conf = sum(r["confidence"] for r in valid) / len(valid)
        avg_rate = sum(r["rate"] for r in valid) / len(valid)
        price = features["close"].iloc[-1]

        reason = []
        try:
            rsi = float(features["rsi"].iloc[-1])
            macd = float(features["macd"].iloc[-1])
            boll = float(features["bollinger"].iloc[-1])
            if final_direction == "롱":
                if rsi < 30: reason.append("RSI 과매도")
                if macd > 0: reason.append("MACD 상승")
            else:
                if rsi > 70: reason.append("RSI 과매수")
                if macd < 0: reason.append("MACD 하락")
            if boll > 1: reason.append("볼린저 상단")
            elif boll < -1: reason.append("볼린저 하단")
        except Exception as e:
            print(f"[지표 예외] {symbol}-{strategy}: {e}")

        return {
            "symbol": symbol, "strategy": strategy, "model": "ensemble",
            "direction": final_direction, "confidence": avg_conf,
            "rate": avg_rate, "price": price,
            "target": price * (1 + avg_rate) if final_direction == "롱" else price * (1 - avg_rate),
            "stop": price * (1 - STOP_LOSS_PCT) if final_direction == "롱" else price * (1 + STOP_LOSS_PCT),
            "reason": ", ".join(reason), "success": True,
            "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return failed_result(symbol, strategy, f"예외 발생: {e}")
