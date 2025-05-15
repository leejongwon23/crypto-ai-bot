import os
import torch
import numpy as np
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from window_optimizer import find_best_window

DEVICE = torch.device("cpu")
STOP_LOSS_PCT = 0.02
MODEL_DIR = "/persistent/models"

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

        model_type = None
        for mt in ["lstm", "cnn_lstm", "transformer"]:
            model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{mt}.pt")
            if os.path.exists(model_path):
                model_type = mt
                break

        if model_type is None:
            print(f"[SKIP] {symbol}-{strategy} → 저장된 모델 없음")
            return None

        model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.pt")
        model = get_model(model_type=model_type, input_size=input_size)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            signal, confidence = model(X_tensor)
            if signal is None or confidence is None:
                print(f"[SKIP] {symbol}-{strategy} {model_type} → None 반환")
                return None
            signal = signal.squeeze().item()
            confidence = confidence.squeeze().item()
            direction = "롱" if signal > 0.5 else "숏"
            rate = abs(signal - 0.5) * 2

        price = features["close"].iloc[-1]
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
            "model": model_type,
            "direction": direction,
            "confidence": confidence,
            "rate": rate,
            "price": price,
            "target": price * (1 + rate) if direction == "롱" else price * (1 - rate),
            "stop": price * (1 - STOP_LOSS_PCT) if direction == "롱" else price * (1 + STOP_LOSS_PCT),
            "reason": ", ".join(reason)
        }

    except Exception as e:
        print(f"[FATAL] {symbol}-{strategy} 예측 실패: {e}")
        return None
