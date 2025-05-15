# --- [필수 import] ---
import os, time, datetime, threading, gc
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

# --- [기존 모델 성능 평가] ---
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

# --- [개별 모델 학습 함수] ---
def train_one_model(symbol, strategy, model_type, input_size=11, batch_size=32, epochs=10, lr=1e-3):
    print(f"[train] {symbol}-{strategy} ({model_type}) 학습 시작")
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
    model = get_model(model_type=model_type, input_size=input_size)
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.pt")

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
    val_loader = DataLoader(val_set, batch_size=batch_size)

    if len(train_loader.dataset) < 2:
        print(f"[스킵] {symbol}-{strategy} 학습 샘플 부족")
        return

    wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=best_window)
    if wrong_data:
        try:
            shapes = set([x[0].shape for x in wrong_data])
            if len(shapes) > 1:
                print(f"[무시됨] {symbol}-{strategy} 오답 시퀀스 shape 불일치: {shapes}")
            else:
                wrong_loader = DataLoader(wrong_data, batch_size=batch_size, shuffle=True)
                for xb, yb in wrong_loader:
                    pred, _ = model(xb)
                    if pred is None:
                        print(f"[오답 무시] {symbol}-{strategy} → None 반환")
                        continue
                    loss = criterion(pred, yb)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
        except Exception as e:
            print(f"[오답 학습 실패] {symbol}-{strategy} → {e}")

    for epoch in range(epochs):
        for xb, yb in train_loader:
            pred, _ = model(xb)
            if pred is None:
                print(f"[학습 무시] {symbol}-{strategy} → None 반환")
                continue
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        y_true, y_pred, y_prob = [], [], []
        for xb, yb in val_loader:
            out, _ = model(xb)
            if out is None:
                print(f"[평가 무시] {symbol}-{strategy} → None 반환")
                continue
            y_prob.extend(out.squeeze().numpy().tolist())
            y_true.extend(yb.numpy().tolist())
            y_pred.extend((out.squeeze().numpy() > 0.5).astype(int).tolist())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logger.log_training_result(symbol, strategy, model_type, acc, f1, log_loss(y_true, y_prob))

    val_X_tensor = torch.stack([xb for xb, _ in val_loader])
    val_y_tensor = torch.tensor(y_true, dtype=torch.float32)
    current_score = acc + f1
    previous_score = evaluate_existing_model(
        model_path,
        get_model(model_type, input_size),
        val_X_tensor.view(len(val_y_tensor), best_window, input_size),
        val_y_tensor
    )

    if current_score > previous_score:
        torch.save(model.state_dict(), model_path)
        print(f"✅ 저장됨: {model_path} (new score: {current_score:.4f} > old: {previous_score:.4f})")
    elif current_score == previous_score:
        print(f"ℹ️ 저장 안함: 기존과 성능 동일 (score: {current_score:.4f})")
    else:
        print(f"❌ 저장 안됨: 기존보다 성능 낮음 (new: {current_score:.4f} < old: {previous_score:.4f})")

    required = len(val_set) + best_window
    flat = feature_dicts[-required:-best_window]
    flat_tensor = torch.tensor([list(row.values()) for row in flat], dtype=torch.float32)
    expected_shape = (len(val_set), best_window, input_size)
    if flat_tensor.numel() == np.prod(expected_shape):
        compute_X_val = flat_tensor.view(*expected_shape)
        compute_y_val = y_tensor[-len(val_set):]
        importances = compute_feature_importance(model, compute_X_val, compute_y_val, list(df_feat.columns))
        save_feature_importance(importances, symbol, strategy, model_type)

# --- [전략별 전체 모델 학습 함수] ---
def train_one_strategy(strategy):
    for symbol in SYMBOLS:
        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            try:
                train_one_model(symbol, strategy, model_type)
            except Exception as e:
                print(f"[오류] {symbol}-{strategy}-{model_type} 학습 실패: {e}")

# --- [전체 전략 학습 함수] ---
def train_all_models():
    for strategy in STRATEGY_GAIN_RANGE:
        train_one_strategy(strategy)

# --- [백그라운드 자동 학습] ---
def background_auto_train():
    def loop(strategy, interval_sec):
        while True:
            for symbol in SYMBOLS:
                for model_type in ["lstm", "cnn_lstm", "transformer"]:
                    try:
                        train_one_model(symbol, strategy, model_type)
                        gc.collect()
                    except Exception as e:
                        print(f"[오류] {symbol}-{strategy}-{model_type} 학습 실패: {e}")
            time.sleep(interval_sec)

    intervals = {"단기": 10800, "중기": 21600, "장기": 43200}
    for strategy, interval in intervals.items():
        threading.Thread(target=loop, args=(strategy, interval), daemon=True).start()

# --- [실행] ---
background_auto_train()
