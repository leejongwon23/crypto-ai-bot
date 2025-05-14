# --- [필수 import] ---
import os, time, datetime, threading, gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss

from data.utils import SYMBOLS, get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from wrong_data_loader import load_wrong_prediction_data
from feature_importance import compute_feature_importance, save_feature_importance
import logger
from window_optimizer import find_best_window

# --- [설정] ---
DEVICE = torch.device("cpu")
STOP_LOSS_PCT = 0.02
PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
WRONG_DIR = os.path.join(PERSIST_DIR, "wrong")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WRONG_DIR, exist_ok=True)

STRATEGY_GAIN_RANGE = {
    "단기": (0.03, 0.50),
    "중기": (0.05, 0.80),
    "장기": (0.10, 1.00)
}

# --- [데이터셋 생성] ---
def create_dataset(features, window):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        if current_close == 0: continue
        change = (future_close - current_close) / current_close
        label = 1 if change > 0 else 0
        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    return np.array(X), np.array(y)

# --- [모델 학습] ---
def train_model(symbol, strategy, input_size=11, batch_size=32, epochs=10, lr=1e-3):
    print(f"[train_model] 시작: {symbol}-{strategy}")
    try:
        best_window = find_best_window(symbol, strategy)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < best_window + 10:
            print(f"❌ {symbol}-{strategy} 데이터 부족")
            return
        df_feat = compute_features(df)
        if len(df_feat) < best_window + 1:
            print(f"❌ {symbol}-{strategy} feature 부족")
            return

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_feat.values)
        feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]
        X, y = create_dataset(feature_dicts, best_window)
        if len(X) < 2:
            print(f"[SKIP] {symbol}-{strategy} 유효 시퀀스 부족 → {len(X)}개")
            return

        input_size = X.shape[2] if len(X.shape) == 3 else X.shape[1]

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type=model_type, input_size=input_size)
            model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.pt")
            if os.path.exists(model_path): os.remove(model_path)

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
                print(f"[스킵] {symbol}-{strategy} 학습 샘플 부족")
                continue

            # 오답 학습
            wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=best_window)
            if wrong_data:
                wrong_loader = DataLoader(wrong_data, batch_size=batch_size, shuffle=True)
                for xb, yb in wrong_loader:
                    pred, _ = model(xb)
                    loss = criterion(pred, yb)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

            # 정규 학습
            for epoch in range(epochs):
                for xb, yb in train_loader:
                    pred, _ = model(xb)
                    loss = criterion(pred, yb)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

            # 평가
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

            # 중요도 분석
            if model_type in ["lstm", "cnn_lstm", "transformer"]:
                required = len(val_set) + best_window
                flat = feature_dicts[-required:-best_window]
                flat_tensor = torch.tensor([list(row.values()) for row in flat], dtype=torch.float32)
                expected_shape = (len(val_set), best_window, input_size)
                if flat_tensor.numel() == np.prod(expected_shape):
                    compute_X_val = flat_tensor.view(*expected_shape)
                    compute_y_val = y_tensor[-len(val_set):]
                    importances = compute_feature_importance(model, compute_X_val, compute_y_val, list(df_feat.columns))
                    save_feature_importance(importances, symbol, strategy, model_type)
                else:
                    print(f"[SKIP] {symbol}-{strategy}-{model_type} 중요도 분석 생략 (view 실패)")

            torch.save(model.state_dict(), model_path)
            print(f"✅ 모델 저장 완료: {model_path}")

    except Exception as e:
        print(f"[FATAL] {symbol}-{strategy} 학습 실패: {e}")

# --- [학습 루프 / 수동 호출] ---
def train_model_loop(strategy):
    print(f"[train_model_loop] {strategy} 전략 전체 학습 루프 시작")
    for symbol in SYMBOLS:
        try:
            train_model(symbol, strategy)
        except Exception as e:
            print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")

def auto_train_all():
    print("[auto_train_all] 전체 전략 수동 학습 시작")
    for strategy in STRATEGY_GAIN_RANGE:
        train_model_loop(strategy)

# --- [주기적 백그라운드 학습 실행] ---
def background_auto_train():
    def loop(strategy, interval_sec):
        while True:
            print(f"[주기적 학습] {strategy} 시작")
            for symbol in SYMBOLS:
                try:
                    train_model(symbol, strategy)
                    gc.collect()
                except Exception as e:
                    print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")
            print(f"[주기적 학습] {strategy} 종료")
            time.sleep(interval_sec)

    strategy_intervals = {
        "단기": 10800,
        "중기": 21600,
        "장기": 43200
    }
    for strategy, interval in strategy_intervals.items():
        threading.Thread(target=loop, args=(strategy, interval), daemon=True).start()

# --- [실행] ---
background_auto_train()
