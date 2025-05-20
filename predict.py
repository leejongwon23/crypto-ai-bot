# --- [필수 import] ---
import os
import torch
import numpy as np
import datetime
import pytz
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from sklearn.metrics import log_loss
from logger import get_min_gain

DEVICE = torch.device("cpu")
STOP_LOSS_PCT = 0.02
MODEL_DIR = "/persistent/models"

# --- ✅ 시간 함수 ---
def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# --- ✅ 실패 결과 반환 구조 ---
def failed_result(symbol, strategy, reason):
    dummy_price = 1.0
    return {
        "symbol": symbol, "strategy": strategy, "success": False,
        "reason": reason, "direction": "롱", "model": "ensemble",
        "confidence": 0.0, "rate": 0.0, "price": dummy_price,
        "target": dummy_price, "stop": dummy_price,
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
    }

# --- ✅ 메인 예측 함수 ---
def predict(symbol, strategy):
    try:
        best_window = find_best_window(symbol, strategy)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < best_window + 1:
            return failed_result(symbol, strategy, "데이터 부족")

        features = compute_features(df)
        if features is None or len(features) < best_window + 1:
            return failed_result(symbol, strategy, "feature 부족")

        try:
            X_raw = features.iloc[-best_window:].values
            if X_raw.shape[0] != best_window:
                raise ValueError("시퀀스 길이 부족")
            X = np.expand_dims(X_raw, axis=0)
            if len(X.shape) != 3:
                raise ValueError("시퀀스 형상 오류")
        except Exception as e:
            return failed_result(symbol, strategy, f"입력 오류: {e}")

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
                    signal = signal.squeeze().item()
                    confidence = confidence.squeeze().item()
                    if not (0.0 <= signal <= 1.0) or 0.495 <= signal <= 0.505:
                        continue

                    direction = "롱" if signal > 0.5 else "숏"
                    raw_rate = ((signal - 0.5) * 2) ** 2  # 제곱 보정
                    rate = raw_rate * min_gain * 0.8

                    weight = get_model_weight(model_type, strategy)
                    fake_y = np.array([1 if signal > 0.5 else 0])
                    fake_p = np.array([signal])
                    try:
                        loss = log_loss(fake_y, fake_p, labels=[0, 1])
                        penalty = max(0.01, 1 - loss)
                    except:
                        penalty = confidence

                    final_conf = (confidence + penalty + weight) / 3
                    score = final_conf * rate * weight

                    results.append({
                        "model": model_type,
                        "direction": direction,
                        "confidence": final_conf,
                        "weight": weight,
                        "score": score,
                        "rate": rate
                    })
            except Exception:
                continue

        if not results:
            return failed_result(symbol, strategy, "모든 모델 예측 실패")

        dir_count = {"롱": 0, "숏": 0}
        for r in results:
            dir_count[r["direction"]] += 1

        if dir_count["롱"] >= 2:
            final_dir = "롱"
        elif dir_count["숏"] >= 2:
            final_dir = "숏"
        elif len(results) == 1:
            final_dir = results[0]["direction"]
        else:
            return failed_result(symbol, strategy, "모델 방향 불일치")

        final_results = [r for r in results if r["direction"] == final_dir]
        avg_conf = sum(r["confidence"] for r in final_results) / len(final_results)
        avg_rate = sum(r["rate"] for r in final_results) / len(final_results)
        price = features["close"].iloc[-1]

        # 보조지표 해석
        rsi = features["rsi"].iloc[-1] if "rsi" in features else 50
        macd = features["macd"].iloc[-1] if "macd" in features else 0
        boll = features["bollinger"].iloc[-1] if "bollinger" in features else 0
        reason = []
        if final_dir == "롱":
            if rsi < 30: reason.append("RSI 과매도")
            if macd > 0: reason.append("MACD 상승 전환")
        else:
            if rsi > 70: reason.append("RSI 과매수")
            if macd < 0: reason.append("MACD 하락 전환")
        if boll > 1: reason.append("볼린저 상단 돌파")
        elif boll < -1: reason.append("볼린저 하단 이탈")

        return {
            "symbol": symbol, "strategy": strategy, "model": "ensemble",
            "direction": final_dir, "confidence": avg_conf, "rate": avg_rate,
            "price": price,
            "target": price * (1 + avg_rate) if final_dir == "롱" else price * (1 - avg_rate),
            "stop": price * (1 - STOP_LOSS_PCT) if final_dir == "롱" else price * (1 + STOP_LOSS_PCT),
            "reason": ", ".join(reason), "success": True,
            "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return failed_result(symbol, strategy, f"예외 발생: {e}")
