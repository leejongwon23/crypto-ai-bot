import os
import time
import datetime
import threading
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features
from model.base_model import get_model, format_prediction
from wrong_data_loader import load_wrong_prediction_data
import logger
from src.message_formatter import format_message
from telegram_bot import send_message
from feature_importance import compute_feature_importance, save_feature_importance
from window_optimizer import find_best_window
import gc

DEVICE = torch.device("cpu")
STOP_LOSS_PCT = 0.02

STRATEGY_GAIN_RANGE = {
    "단기": (0.03, 0.50),
    "중기": (0.05, 0.80),
    "장기": (0.10, 1.00)
}

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
WRONG_DIR = os.path.join(PERSIST_DIR, "wrong")
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "train_log.txt")
PRED_LOG_FILE = os.path.join(PERSIST_DIR, "prediction_log.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(WRONG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def create_dataset(features, strategy, window=20):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        if current_close == 0:
            continue
        change = (future_close - current_close) / current_close
        label = 1 if change > 0 else 0
        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    return np.array(X), np.array(y)

def train_model(symbol, strategy, input_size=11, batch_size=32, epochs=10, lr=1e-3):
    print(f"[train_model] 시작: {symbol}-{strategy}")
    best_window = find_best_window(symbol, strategy)
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < best_window + 10:
        print(f"❌ {symbol}-{strategy} 데이터 부족")
        return
    df_feat = compute_features(df)
    if len(df_feat) < best_window + 1:
        return
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]
    X, y = create_dataset(feature_dicts, strategy, window=best_window)
    if len(X) == 0:
        print(f"⚠️ {symbol}-{strategy} 유효 시퀀스 없음")
        return
    input_size = X.shape[2] if len(X.shape) == 3 else X.shape[1]
    for model_type in ["lstm", "cnn_lstm", "transformer"]:
        model = get_model(model_type=model_type, input_size=input_size)
        model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.pt")
        if os.path.exists(model_path):
            os.remove(model_path)
        model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        val_len = int(len(dataset) * 0.2)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        if len(train_loader.dataset) < 2:
            print(f"[스킵] {symbol}-{strategy} 데이터 부족 → 샘플 수: {len(train_loader.dataset)}")
            continue

        wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=best_window)
        if wrong_data:
            wrong_loader = DataLoader(wrong_data, batch_size=batch_size, shuffle=True)
            for xb, yb in wrong_loader:
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ✅ 성능 평가 및 기록
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(val_set, batch_size=batch_size)
            y_true, y_pred, y_prob = [], [], []
            for xb, yb in val_loader:
                out, _ = model(xb)
                y_prob.extend(out.squeeze().numpy().tolist())
                y_true.extend(yb.numpy().tolist())
                y_pred.extend((out.squeeze().numpy() > 0.5).astype(int).tolist())
            if len(y_true) >= 1:
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                loss = log_loss(y_true, y_prob)
                logger.log_training_result(symbol, strategy, model_type, acc, f1, loss)

        # ✅ 중요도 분석
        if model_type == "lstm":
            feature_names = list(df_feat.columns)
            compute_X_val = torch.tensor(
                [list(row.values()) for row in feature_dicts[-(len(val_set)+best_window):-(best_window)]],
                dtype=torch.float32
            ).view(len(val_set), best_window, input_size)
            compute_y_val = y_tensor[-len(val_set):]
            importances = compute_feature_importance(model, compute_X_val, compute_y_val, feature_names)
            save_feature_importance(importances, symbol, strategy, model_type)

        torch.save(model.state_dict(), model_path)
        print(f"✅ 모델 저장됨: {model_path}")

def auto_train_all():
    print("[auto_train_all] 전체 코인 및 전략 학습 시작")
    for strategy in STRATEGY_GAIN_RANGE:
        for symbol in SYMBOLS:
            try:
                train_model(symbol, strategy)
            except Exception as e:
                print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")

def background_auto_train():
    def loop(strategy, interval_sec):
        while True:
            print(f"[전략별 학습 시작] → {strategy}")
            for symbol in SYMBOLS:
                try:
                    train_model(symbol, strategy)
                    gc.collect()
                except Exception as e:
                    print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")
            print(f"[전략별 학습 종료] → {strategy}")
            time.sleep(interval_sec)

    strategy_intervals = {
        "단기": 10800,
        "중기": 21600,
        "장기": 43200
    }

    for strategy, interval in strategy_intervals.items():
        threading.Thread(target=loop, args=(strategy, interval), daemon=True).start()

def predict(symbol, strategy):
    from window_optimizer import find_best_window
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
            print(f"[오류] {model_path} 모델 구조 불일치 → 삭제 후 재학습 대기")
            os.remove(model_path)
            continue

        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            signal, confidence = model(X_tensor)
            signal = signal.squeeze().item()
            confidence = confidence.squeeze().item()
            rate = abs(signal - 0.5) * 2
            direction = "롱" if signal > 0.5 else "숏"
            results.append({
                "model": model_type,
                "symbol": symbol,
                "strategy": strategy,
                "confidence": confidence,
                "rate": rate,
                "direction": direction
            })

    if not results:
        return None

    direction_votes = [r["direction"] for r in results]
    final_direction = max(set(direction_votes), key=direction_votes.count)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_rate = sum(r["rate"] for r in results) / len(results)
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
        "direction": final_direction,
        "confidence": avg_confidence,
        "rate": avg_rate,
        "price": price,
        "target": price * (1 + avg_rate) if final_direction == "롱" else price * (1 - avg_rate),
        "stop": price * (1 - STOP_LOSS_PCT) if final_direction == "롱" else price * (1 + STOP_LOSS_PCT),
        "reason": ", ".join(reason)
    }

def main():
    logger.evaluate_predictions(get_price_now)
    for strategy in STRATEGY_GAIN_RANGE:
        results = []
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
        if results:
            logger.log_prediction(
                symbol=results[0]["symbol"],
                strategy=results[0]["strategy"],
                direction=results[0]["direction"],
                entry_price=results[0]["price"],
                target_price=results[0]["target"],
                timestamp=datetime.datetime.utcnow().isoformat(),
                confidence=results[0]["confidence"]
            )
            msg = format_message(results[0])
            send_message(msg)

def get_price_now(symbol):
    from data.utils import get_realtime_prices
    prices = get_realtime_prices()
    return prices.get(symbol)

background_auto_train()
