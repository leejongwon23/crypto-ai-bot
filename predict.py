# --- [필수 import] ---
import os
import torch
import numpy as np
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window

# --- [설정] ---
DEVICE = torch.device("cpu")
STOP_LOSS_PCT = 0.02
MODEL_DIR = "/persistent/models"

# --- [예측 함수] ---
def predict(symbol, strategy):
    try:
        best_window = find_best_window(symbol, strategy)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < best_window + 1:
            return None

        features = compute_features(df)
        if len(features) < best_window + 1:
            return None

        X = features.iloc[-best_window:].values
        X = np.expand_dims(X, axis=0)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        input_size = X.shape[2] if len(X.shape) == 3 else X.shape[1]

        results = []
        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type=model_type, input_size=input_size)
            model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.pt")

            if not os.path.exists(model_path):
                continue

            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            except RuntimeError:
                print(f"[SKIP] {model_type} 모델 로드 실패 → {symbol}-{strategy}")
                continue

            model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                signal, confidence = model(X_tensor)
                signal = signal.squeeze().item()
                confidence = confidence.squeeze().item()
                direction = "롱" if signal > 0.5 else "숏"
                weight = get_model_weight(model_type, strategy)
                score = confidence * weight
                rate = abs(signal - 0.5) * 2

                results.append({
                    "model": model_type,
                    "symbol": symbol,
                    "strategy": strategy,
                    "confidence": confidence,
                    "weight": weight,
                    "score": score,
                    "rate": rate,
                    "direction": direction
                })

        if not results:
            return None

        long_score = sum(r["score"] for r in results if r["direction"] == "롱")
        short_score = sum(r["score"] for r in results if r["direction"] == "숏")
        final_direction = "롱" if long_score > short_score else "숏"
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_rate = sum(r["rate"] for r in results) / len(results)
        price = features["close"].iloc[-1]

        # 설명용 기술 지표 해석
        rsi = features["rsi"].iloc[-1] if "rsi" in features else 50
        macd = features["macd"].iloc[-1] if "macd" in features else 0
        boll = features["bollinger"].iloc[-1] if "bollinger" in features else 0
        reason = []
        if rsi < 30: reason.append("RSI 과매도")
        elif rsi > 70: reason.append("RSI 과매수")
        reason.append("MACD 상승 전환" if macd > 0 else "MACD 하락 전환")
        if boll > 1: reason.append("볼린저 상단 돌파")
        elif boll < -1: reason.append("볼린저 하단 이탈")

        return {
            "symbol": symbol,
            "strategy": strategy,
            "direction": final_direction,
            "confidence": avg_confidence,
            "rate": avg_rate,
            "price": price,
            "target": price * (1 + avg_rate) if final_direction == "롱" else price * (1 - avg_rate),
            "stop": price * (1 - STOP_LOSS_PCT) if final_direction == "롱" else price * (1 + STOP_LOSS_PCT),
            "reason": ", ".join(reason)
        }

    except Exception as e:
        print(f"[FATAL] {symbol}-{strategy} 예측 실패: {e}")
        return None
