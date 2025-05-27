import os, torch, numpy as np, pandas as pd, datetime, pytz, sys
from sklearn.preprocessing import MinMaxScaler
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
    is_volatility = "_v" in symbol
    try:
        log_prediction(symbol, strategy, direction="롱", entry_price=0, target_price=0,
                       model="ensemble", success=False, reason=reason,
                       rate=0.0, timestamp=t, volatility=is_volatility)
    except Exception as e:
        print(f"[경고] log_prediction 실패: {e}")
        sys.stdout.flush()
    return {
        "symbol": symbol, "strategy": strategy, "success": False, "reason": reason,
        "direction": "롱", "model": "ensemble", "rate": 0.0,
        "price": 1.0, "target": 1.0, "stop": 1.0, "timestamp": t
    }

def predict(symbol, strategy):
    try:
        print(f"[PREDICT] {symbol}-{strategy} 시작")
        sys.stdout.flush()
        is_volatility = "_v" in symbol
        window = find_best_window(symbol, strategy)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < window + 1:
            return failed_result(symbol, strategy, "데이터 부족")
        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < window + 1:
            return failed_result(symbol, strategy, "feature 부족")
        feat = pd.DataFrame(MinMaxScaler().fit_transform(feat.dropna()), columns=feat.columns)
        X = np.expand_dims(feat.iloc[-window:].values, axis=0)
        if X.shape[1] != window:
            return failed_result(symbol, strategy, "시퀀스 형상 오류")

        model_files = {}
        for f in os.listdir(MODEL_DIR):
            if not f.endswith(".pt"): continue
            parts = f.replace(".pt", "").split("_")
            if len(parts) < 3: continue
            f_sym, f_strat, f_type = parts[0], parts[1], "_".join(parts[2:])
            if f_sym == symbol and f_strat == strategy:
                model_files[f_type] = os.path.join(MODEL_DIR, f)

        if not model_files:
            return failed_result(symbol, strategy, "모델 없음")

        rates = []
        for model_type, path in model_files.items():
            try:
                model = get_model(model_type, X.shape[2])
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    output = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                    if isinstance(output, tuple): output = output[0]
                    rate = float(output.squeeze())
                    if np.isnan(rate) or rate < 0: continue
                    rates.append(rate)
            except Exception as e:
                print(f"[모델 예측 실패] {symbol}-{strategy}-{model_type}: {e}")
                sys.stdout.flush()

        if not rates:
            return failed_result(symbol, strategy, "모든 모델 예측 실패")

        avg_rate = np.mean(rates)
        if avg_rate < MIN_EXPECTED_RATES.get(strategy, 0.01):
            return failed_result(symbol, strategy, f"예측 수익률 기준 미달 ({avg_rate:.4f})")

        price = feat["close"].iloc[-1]
        if np.isnan(price):
            return failed_result(symbol, strategy, "price NaN 발생")

        t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
        log_prediction(symbol, strategy, "롱", entry_price=price,
                       target_price=price * (1 + avg_rate),
                       model="ensemble", success=True,
                       reason="수익률 예측 성공", rate=avg_rate,
                       timestamp=t, volatility=is_volatility)

        return {
            "symbol": symbol, "strategy": strategy, "model": "ensemble", "direction": "롱",
            "rate": avg_rate, "price": price,
            "target": price * (1 + avg_rate),
            "stop": price * (1 - STOP_LOSS_PCT),
            "reason": "수익률 예측 성공", "success": True, "timestamp": t
        }

    except Exception as e:
        print(f"[예외] 예측 실패: {symbol}-{strategy} → {e}")
        sys.stdout.flush()
        return failed_result(symbol, strategy, f"예외 발생: {e}")
