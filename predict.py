import os, torch, numpy as np, pandas as pd, datetime, pytz, sys
from sklearn.preprocessing import MinMaxScaler
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from window_optimizer import find_best_window
from logger import log_prediction

DEVICE, MODEL_DIR = torch.device("cpu"), "/persistent/models"
STOP_LOSS_PCT = 0.02
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def failed_result(symbol, strategy, model_type, reason, source="일반"):
    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
    is_volatility = "_v" in symbol
    try:
        log_prediction(symbol, strategy, direction="예측실패", entry_price=0, target_price=0,
                       model=model_type, success=False, reason=reason,
                       rate=0.0, timestamp=t, volatility=is_volatility, source=source)
    except Exception as e:
        print(f"[경고] log_prediction 실패: {e}")
        sys.stdout.flush()
    return {
        "symbol": symbol, "strategy": strategy, "success": False, "reason": reason,
        "direction": "예측실패", "model": model_type, "rate": 0.0,
        "price": 0.0, "target": 0.0, "stop": 0.0, "timestamp": t,
        "source": source
    }

def predict(symbol, strategy, source="일반"):
    try:
        print(f"[PREDICT] {symbol}-{strategy} 시작")
        sys.stdout.flush()
        is_volatility = "_v" in symbol
        window = find_best_window(symbol, strategy)

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < window + 1:
            return [failed_result(symbol, strategy, "unknown", "데이터 부족", source=source)]

        raw_close = df['close'].iloc[-1]
        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.dropna().shape[0] < window + 1:
            return [failed_result(symbol, strategy, "unknown", "feature 부족", source=source)]

        # ✅ timestamp 보존 → MinMaxScaler 적용
        if "timestamp" not in feat.columns:
            return [failed_result(symbol, strategy, "unknown", "timestamp 없음", source=source)]

        raw_feat = feat.copy()
        timestamps = raw_feat["timestamp"]
        feat_scaled = MinMaxScaler().fit_transform(raw_feat.drop(columns=["timestamp"]))
        feat = pd.DataFrame(feat_scaled, columns=raw_feat.drop(columns=["timestamp"]).columns)
        feat["timestamp"] = timestamps.values

        # ✅ 시퀀스 길이 및 차원 확인
        if feat.shape[0] < window:
            return [failed_result(symbol, strategy, "unknown", "시퀀스 부족", source=source)]

        X = np.expand_dims(feat.iloc[-window:].drop(columns=["timestamp"]).values, axis=0)
        if X.shape[1] != window or len(X.shape) != 3:
            return [failed_result(symbol, strategy, "unknown", "시퀀스 형상 오류", source=source)]

        model_files = {}
        for f in os.listdir(MODEL_DIR):
            if not f.endswith(".pt"):
                continue
            parts = f.replace(".pt", "").split("_")
            if len(parts) < 3:
                continue
            f_type = parts[-1]
            f_strat = parts[-2]
            f_sym = "_".join(parts[:-2])
            if f_sym == symbol and f_strat == strategy:
                model_files[f_type] = os.path.join(MODEL_DIR, f)

        if not model_files:
            return [failed_result(symbol, strategy, "unknown", "모델 없음", source=source)]

        horizon_map = {"단기": 4, "중기": 24, "장기": 168}
        target_hours = horizon_map.get(strategy, 4)
        period_label = f"{target_hours}h"

        predictions = []
        for model_type, path in model_files.items():
            try:
                model = get_model(model_type, X.shape[2])
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()

                with torch.no_grad():
                    output = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
                    if isinstance(output, tuple):
                        output = output[0]
                    raw_rate = float(output.squeeze())

                    if np.isnan(raw_rate):
                        predictions.append(failed_result(symbol, strategy, model_type, "NaN 예측값", source=source))
                        continue
                    if np.isnan(raw_close):
                        predictions.append(failed_result(symbol, strategy, model_type, "price NaN 발생", source=source))
                        continue

                    if raw_rate >= 0:
                        direction = "롱"
                        rate = raw_rate
                        target = raw_close * (1 + rate)
                        stop = raw_close * (1 - STOP_LOSS_PCT)
                    else:
                        direction = "숏"
                        rate = -raw_rate
                        target = raw_close * (1 - rate)
                        stop = raw_close * (1 + STOP_LOSS_PCT)

                    t = now_kst().strftime("%Y-%m-%d %H:%M:%S")
                    success = True

                    log_prediction(
                        symbol=symbol, strategy=strategy, direction=direction,
                        entry_price=raw_close, target_price=target,
                        model=model_type, success=success,
                        reason=f"{period_label} 예측 완료",
                        rate=rate, timestamp=t,
                        volatility=is_volatility, source=source
                    )

                    predictions.append({
                        "symbol": symbol, "strategy": strategy, "model": model_type,
                        "direction": direction, "rate": rate, "price": raw_close,
                        "target": target, "stop": stop, "reason": f"{period_label} 예측 완료",
                        "success": success, "timestamp": t, "source": source
                    })

            except Exception as e:
                predictions.append(failed_result(symbol, strategy, model_type, f"예측 예외: {e}", source=source))

        return predictions if predictions else [failed_result(symbol, strategy, "unknown", "모든 모델 예측 실패", source=source)]

    except Exception as e:
        return [failed_result(symbol, strategy, "unknown", f"예외 발생: {e}", source=source)]
