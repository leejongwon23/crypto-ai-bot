import os
import time
import datetime
import threading
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features
from model.base_model import get_model, format_prediction
from wrong_data_loader import load_wrong_prediction_data
import logger
from src.message_formatter import format_message
from telegram_bot import send_message
import gc

DEVICE = torch.device("cpu")
WINDOW = 30
STOP_LOSS_PCT = 0.02

STRATEGY_GAIN_RANGE = {
    "단기": (0.03, 0.50),
    "중기": (0.05, 0.80),
    "장기": (0.10, 1.00)
}

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "train_log.txt")

def create_dataset(features, strategy, window=30):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        if current_close == 0:
            print(f"[무시됨] current_close = 0 at index {i}")
            continue
        change = (future_close - current_close) / current_close
        min_gain = STRATEGY_GAIN_RANGE[strategy][0]
        if abs(change) < min_gain or abs(change) > 1.0:
            print(f"[무시됨] change={change:.4f}가 조건 불충족 (min_gain={min_gain})")
            continue
        label = 1 if change > 0 else 0
        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    print(f"[create_dataset] 최종 생성된 학습데이터 개수: {len(X)}")
    return np.array(X), np.array(y)

def train_model(symbol, strategy, input_size=11, batch_size=32, epochs=10, lr=1e-3):
    print(f"[train_model] 시작: {symbol}-{strategy}")
    df = get_kline_by_strategy(symbol, strategy)
    if df is None:
        print(f"❌ {symbol}-{strategy} 수집된 원시 데이터 없음: None", flush=True)
        return
    if len(df) < WINDOW + 10:
        print(f"❌ {symbol}-{strategy} 수집된 원시 데이터 너무 짧음: {len(df)}개", flush=True)
        return
    print(f"✅ {symbol}-{strategy} 원시 데이터 수집 성공: {len(df)}행")
    df_feat = compute_features(df)
    if len(df_feat) < WINDOW + 1:
        print(f"❌ {symbol}-{strategy} 특징 추출 후 데이터 부족: {len(df_feat)}개", flush=True)
        return
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]
    X, y = create_dataset(feature_dicts, strategy, window=WINDOW)
    print(f"▶️ {symbol}-{strategy} 데이터 개수: X={len(X)}, y={len(y)}", flush=True)
    if len(X) == 0:
        print(f"⚠️ {symbol}-{strategy} 학습 안 됨: 유효 시퀀스 없음", flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.datetime.utcnow()}] ❌ {symbol}-{strategy} 학습 실패 (데이터 없음)\n")
        return
    os.makedirs("models", exist_ok=True)
    for model_type in ["lstm", "cnn_lstm", "transformer"]:
        model = get_model(model_type=model_type, input_size=input_size)
        model_path = f"models/{symbol}_{strategy}_{model_type}.pt"
        if os.path.exists(model_path):
            print(f"⚠️ {model_path} 기존 모델 삭제 후 재학습합니다.", flush=True)
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
        wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=WINDOW)
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
        torch.save(model.state_dict(), model_path)
        print(f"✅ 모델 저장됨: {model_path}", flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.datetime.utcnow()}] ✅ 저장됨: {model_path}\n")
    print("📁 models 폴더 내용:")
    for file in os.listdir("models"):
        print(" -", file)

def auto_train_all():
    print("[auto_train_all] 전체 코인 및 전략 학습 시작")
    for strategy in STRATEGY_GAIN_RANGE:
        for symbol in SYMBOLS:
            try:
                print(f"[학습 중] {symbol} - {strategy}")
                train_model(symbol, strategy)
            except Exception as e:
                print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")

def predict(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < WINDOW + 1:
        return None
    features = compute_features(df)
    if len(features) < WINDOW + 1:
        return None
    X = features.iloc[-WINDOW:].values
    X = np.expand_dims(X, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    results = []
    for model_type in ["lstm", "cnn_lstm", "transformer"]:
        model = get_model(model_type=model_type, input_size=X.shape[2])
        model_path = f"models/{symbol}_{strategy}_{model_type}.pt"
        if not os.path.exists(model_path):
            return None
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            signal, confidence = model(X_tensor)
            signal = signal.squeeze().item()
            confidence = confidence.squeeze().item()
            rate = abs(signal - 0.5) * 2
            result = format_prediction(signal, confidence, rate)
            result.update({"model": model_type})
            results.append(result)
    if not (results[0]["direction"] == results[1]["direction"] == results[2]["direction"]):
        return None
    filtered = [r for r in results if r["rate"] >= STRATEGY_GAIN_RANGE[strategy][0]]
    if not filtered:
        return None
    top = sorted(filtered, key=lambda x: (x["confidence"], x["rate"], ["cnn_lstm", "lstm", "transformer"].index(x["model"]), x["symbol"]), reverse=True)[0]
    df_close = df["close"].iloc[-1]
    if top["rate"] > STRATEGY_GAIN_RANGE[strategy][1]:
        top["rate"] = STRATEGY_GAIN_RANGE[strategy][1]
    if top["direction"] == "롱":
        top["target"] = df_close * (1 + top["rate"])
        top["stop"] = df_close * (1 - STOP_LOSS_PCT)
    else:
        top["target"] = df_close * (1 - top["rate"])
        top["stop"] = df_close * (1 + STOP_LOSS_PCT)
    top["price"] = df_close
    rsi = features["rsi"].iloc[-1]
    macd = features["macd"].iloc[-1]
    boll = features["bollinger"].iloc[-1]
    reason = []
    if rsi < 30: reason.append("RSI 과매도")
    elif rsi > 70: reason.append("RSI 과매수")
    reason.append("MACD 상승 전환" if macd > 0 else "MACD 하락 전환")
    if boll > 1: reason.append("볼린저 상단 돌파")
    elif boll < -1: reason.append("볼린저 하단 이탈")
    top["reason"] = ", ".join(reason)
    return top

def background_auto_train(interval_sec=21600):
    strategies = list(STRATEGY_GAIN_RANGE.keys())
    idx = 0
    def loop():
        nonlocal idx
        while True:
            current_strategy = strategies[idx % len(strategies)]
            print(f"[전략별 학습 시작] → {current_strategy}")
            for symbol in SYMBOLS:
                try:
                    train_model(symbol, current_strategy)
                    gc.collect()
                except Exception as e:
                    print(f"[오류] {symbol}-{current_strategy} 학습 실패: {e}")
            idx += 1
            print(f"[전략별 학습 종료] → {current_strategy}, 다음 전략 대기 중...")
            time.sleep(interval_sec)
    t = threading.Thread(target=loop, daemon=True)
    t.start()

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
