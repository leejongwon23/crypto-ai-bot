import os, torch, numpy as np, pandas as pd, datetime, pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import get_min_gain, log_prediction

DEVICE, MODEL_DIR = torch.device("cpu"), "/persistent/models"
STOP_LOSS_PCT = 0.02
MIN_EXPECTED_RATES = {"단기": 0.007, "중기": 0.015, "장기": 0.03}
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def failed_result(symbol, strategy, reason):
    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
    log_prediction(symbol, strategy, direction="롱", entry_price=0, target_price=0,
                   confidence=0, model="ensemble", success=False, reason=reason, rate=0.0, timestamp=t)
    return {
        "symbol": symbol, "strategy": strategy, "success": False, "reason": reason,
        "direction": "롱", "model": "ensemble", "confidence": 0.0, "rate": 0.0,
        "price": 1.0, "target": 1.0, "stop": 1.0, "timestamp": t
    }

def predict(symbol, strategy):
    try:
        window = find_best_window(symbol, strategy)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < window + 1:
            return failed_result(symbol, strategy, "데이터 부족")

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < window + 1:
            return failed_result(symbol, strategy, "feature 부족")

        feat = pd.DataFrame(MinMaxScaler().fit_transform(feat.dropna()), columns=feat.columns)
        X = np.expand_dims(feat.iloc[-window:].values, axis=0)
        if X.shape != (1, window, X.shape[2]):
            return failed_result(symbol, strategy, "시퀀스 형상 오류")

        model_files = {
            f.replace(f"{symbol}_{strategy}_", "").replace(".pt", ""): os.path.join(MODEL_DIR, f)
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pt") and f.startswith(f"{symbol}_{strategy}_")
        }
        if not model_files:
            return failed_result(symbol, strategy, "모델 없음")

        min_gain = get_min_gain(symbol, strategy)
        results = []
        for model_type, path in model_files.items():
            try:
                model = get_model(model_type, X.shape[2])
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    s, c = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                    if s is None or c is None: continue
                    s, c = float(s.squeeze()), float(c.squeeze())
                    if not (0 <= s <= 1) or 0.48 <= s <= 0.52: continue
                    dir = "롱" if s > 0.5 else "숏"
                    raw_rate = abs(s - 0.5) * 2
                    weight = get_model_weight(model_type, strategy)
                    try:
                        penalty = max(0.1, 1 - log_loss([1 if s > 0.5 else 0], [np.clip(s, 1e-6, 1 - 1e-6)], labels=[0, 1]))
                    except: penalty = c
                    conf = (c + penalty) / 2
                    rate = raw_rate * min_gain * conf * (1.2 if strategy in ["단기", "중기"] else 1.4)
                    results.append({
                        "model": model_type, "direction": dir, "confidence": conf,
                        "weight": weight, "score": conf * weight * rate, "rate": rate
                    })
            except Exception as e:
                print(f"[모델 예측 실패] {symbol}-{strategy}-{model_type}: {e}")
        if not results:
            return failed_result(symbol, strategy, "모든 모델 예측 실패")

        dir_counts = {"롱": 0, "숏": 0}
        for r in results: dir_counts[r["direction"]] += 1
        if dir_counts["롱"] >= 2: direction = "롱"
        elif dir_counts["숏"] >= 2: direction = "숏"
        elif len(results) == 1: direction = results[0]["direction"]
        else: return failed_result(symbol, strategy, "모델 방향 불일치")

        final = [r for r in results if r["direction"] == direction]
        conf, rate = np.mean([r["confidence"] for r in final]), np.mean([r["rate"] for r in final])
        if rate < MIN_EXPECTED_RATES.get(strategy, 0.01) and max(r["rate"] for r in final) < MIN_EXPECTED_RATES.get(strategy, 0.01) * 1.2:
            return failed_result(symbol, strategy, f"예측 수익률 기준 미달 ({rate:.4f})")

        price = feat["close"].iloc[-1]
        if np.isnan(price):
            return failed_result(symbol, strategy, "price NaN 발생")

        reason = []
        try:
            rsi, macd, boll = map(float, (feat["rsi"].iloc[-1], feat["macd"].iloc[-1], feat["bollinger"].iloc[-1]))
            if direction == "롱":
                if rsi < 30: reason.append("RSI 과매도")
                if macd > 0: reason.append("MACD 상승")
            else:
                if rsi > 70: reason.append("RSI 과매수")
                if macd < 0: reason.append("MACD 하락")
            reason.append("볼린저 상단" if boll > 1 else "볼린저 하단" if boll < -1 else "")
        except Exception as e:
            print(f"[지표 예외] {symbol}-{strategy}: {e}")

        t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
        log_prediction(symbol, strategy, direction, entry_price=price,
                       target_price=price * (1 + rate) if direction == "롱" else price * (1 - rate),
                       confidence=conf, model="ensemble", success=True, reason=", ".join(reason), rate=rate, timestamp=t)

        return {
            "symbol": symbol, "strategy": strategy, "model": "ensemble", "direction": direction,
            "confidence": conf, "rate": rate, "price": price,
            "target": price * (1 + rate) if direction == "롱" else price * (1 - rate),
            "stop": price * (1 - STOP_LOSS_PCT) if direction == "롱" else price * (1 + STOP_LOSS_PCT),
            "reason": ", ".join([r for r in reason if r]), "success": True,
            "timestamp": t
        }

    except Exception as e:
        return failed_result(symbol, strategy, f"예외 발생: {e}")
