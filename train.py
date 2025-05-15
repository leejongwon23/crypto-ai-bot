# --- [필수 import] ---
import os, time, threading, gc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss

from data.utils import SYMBOLS, get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from wrong_data_loader import load_wrong_prediction_data
from feature_importance import compute_feature_importance, save_feature_importance, drop_low_importance_features
import logger
from window_optimizer import find_best_window

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
    "중기": (0.06, 0.80),
    "장기": (0.10, 1.00)
}

def create_dataset(features, window):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        if any(len(row.values()) != len(features[0].values()) for row in x_seq):
            continue
        current_close = features[i+window-1]['close']
        future_close = features[i+window]['close']
        if current_close == 0:
            continue
        change = (future_close - current_close) / current_close
        label = 1 if change > 0 else 0
        X.append([list(row.values()) for row in x_seq])
        y.append(label)
    if not X:
        return np.array([]), np.array([])
    seq_lens = [len(x) for x in X]
    mode_len = max(set(seq_lens), key=seq_lens.count)
    filtered = [(x, l) for x, l in zip(X, y) if len(x) == mode_len]
    if not filtered:
        return np.array([]), np.array([])
    X, y = zip(*filtered)
    return np.array(X), np.array(y)

def evaluate_existing_model(model_path, model, X_val, y_val):
    if not os.path.exists(model_path): return -1
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            out, _ = model(X_val)
            y_prob = out.squeeze().numpy()
            y_pred = (y_prob > 0.5).astype(int)
            y_true = y_val.numpy()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            return acc + f1
    except:
        return -1

def train_one_model(symbol, strategy, input_size=11, batch_size=32, epochs=10, lr=1e-3):
    print(f"[train] {symbol}-{strategy} 전체 모델 학습 시작")
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
    X_raw, y_raw = create_dataset(feature_dicts, best_window)
    if len(X_raw) < 2:
        print(f"[SKIP] {symbol}-{strategy} 유효 시퀀스 부족 → {len(X_raw)}개")
        return

    input_size = X_raw.shape[2]
    val_len = int(len(X_raw) * 0.2)
    val_X_tensor = torch.tensor(X_raw[-val_len:], dtype=torch.float32)
    val_y_tensor = torch.tensor(y_raw[-val_len:], dtype=torch.float32)

    best_score = -1
    best_model_type = None
    best_model_obj = None

    for model_type in ["lstm", "cnn_lstm", "transformer"]:
        model = get_model(model_type=model_type, input_size=input_size)
        model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(torch.tensor(X_raw, dtype=torch.float32),
                                torch.tensor(y_raw, dtype=torch.float32))
        val_len = int(len(dataset) * 0.2)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        # --- 오답 학습 ---
        wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=best_window)
        if wrong_data:
            try:
                shapes = set([x[0].shape for x in wrong_data])
                mode_shape = max(shapes, key=lambda s: [x[0].shape for x in wrong_data].count(s))
                filtered = [(x, y) for x, y in wrong_data if x.shape == mode_shape]
                if len(filtered) > 1:
                    wrong_loader = DataLoader(filtered, batch_size=batch_size, shuffle=True)
                    for xb, yb in wrong_loader:
                        pred, _ = model(xb)
                        if pred is not None:
                            loss = criterion(pred, yb)
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
            except Exception as e:
                print(f"[오답 학습 실패] {symbol}-{strategy} → {e}")

        # --- 정규 학습 ---
        for epoch in range(epochs):
            for xb, yb in train_loader:
                pred, _ = model(xb)
                if pred is None:
                    continue
                loss = criterion(pred, yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        # --- 평가 ---
        model.eval()
        with torch.no_grad():
            out, _ = model(val_X_tensor)
            y_prob = out.squeeze().numpy()
            y_pred = (y_prob > 0.5).astype(int)
            y_true = val_y_tensor.numpy()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            score = acc + f1
            logger.log_training_result(symbol, strategy, model_type, acc, f1, log_loss(y_true, y_prob))

            if score > best_score:
                best_score = score
                best_model_type = model_type
                best_model_obj = model

    if best_model_obj:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{best_model_type}.pt")
        torch.save(best_model_obj.state_dict(), model_path)
        print(f"✅ Best 모델 저장됨: {model_path} (score: {best_score:.4f})")

        # 중요도 분석
        compute_X_val = val_X_tensor
        compute_y_val = val_y_tensor
        importances = compute_feature_importance(best_model_obj, compute_X_val, compute_y_val, list(df_feat.columns))
        save_feature_importance(importances, symbol, strategy, best_model_type)

def train_one_strategy(strategy):
    for symbol in SYMBOLS:
        try:
            train_one_model(symbol, strategy)
        except Exception as e:
            print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")

def train_all_models():
    for strategy in STRATEGY_GAIN_RANGE:
        train_one_strategy(strategy)

def background_auto_train():
    def loop(strategy, interval_sec):
        while True:
            for symbol in SYMBOLS:
                try:
                    train_one_model(symbol, strategy)
                    gc.collect()
                except Exception as e:
                    print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")
            time.sleep(interval_sec)

    intervals = {"단기": 10800, "중기": 21600, "장기": 43200}
    for strategy, interval in intervals.items():
        threading.Thread(target=loop, args=(strategy, interval), daemon=True).start()

background_auto_train()
