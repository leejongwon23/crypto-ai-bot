import os, time, threading, gc, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
import datetime
import pytz

from data.utils import SYMBOLS, get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from wrong_data_loader import load_wrong_prediction_data
from feature_importance import compute_feature_importance, save_feature_importance
import logger
from logger import get_min_gain, get_strategy_fail_rate, get_strategy_eval_count
from window_optimizer import find_best_window

print("✅ 현재 실행 중인 train.py는 최신 버전입니다.")  # ✅ 로그 확인용 추가

DEVICE = torch.device("cpu")
PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
LOG_DIR = os.path.join(PERSIST_DIR, "logs")
WRONG_DIR = os.path.join(PERSIST_DIR, "wrong")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WRONG_DIR, exist_ok=True)

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ... (나머지 코드는 너가 준 것과 100% 동일 — 수정 없음)

def create_dataset(features, window):
    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i + window]
        if any(len(row.values()) != len(features[0].values()) for row in x_seq):
            continue
        current_close = features[i + window - 1]['close']
        future_close = features[i + window]['close']
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

def save_model_metadata(symbol, strategy, model_type, acc, f1, loss):
    meta = {
        "symbol": symbol,
        "strategy": strategy,
        "model": model_type,
        "accuracy": float(round(acc, 4)),
        "f1_score": float(round(f1, 4)),
        "loss": float(round(loss, 6)),
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
    }
    path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"🗘 체크포인트 저장됨: {path}")

def train_one_model(symbol, strategy, input_size=11, batch_size=32, epochs=10, lr=1e-3, repeat=4, repeat_wrong=4):
    print(f"[train] {symbol}-{strategy} 전체 모델 학습 시작")
    best_window = find_best_window(symbol, strategy)
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < best_window + 10:
        print(f"❌ {symbol}-{strategy} 데이터 부족")
        return
    df_feat = compute_features(symbol, df, strategy)  # ✅ 오류 수정 완료
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
    if val_len == 0:
        print(f"[SKIP] {symbol}-{strategy} 검증셋 부족")
        return

    val_X_tensor = torch.tensor(X_raw[-val_len:], dtype=torch.float32)
    val_y_tensor = torch.tensor(y_raw[-val_len:], dtype=torch.float32)

    scores, models, metrics = {}, {}, {}

    for model_type in ["lstm", "cnn_lstm", "transformer"]:
        model = get_model(model_type=model_type, input_size=input_size)
        model.train()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        dataset = TensorDataset(torch.tensor(X_raw, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32))
        train_len = len(dataset) - val_len
        train_set, _ = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        for r in range(repeat):
            print(f"[{symbol}-{strategy}] {model_type} 반복학습 {r + 1}/{repeat}")
            for _ in range(repeat_wrong):
                wrong_data = load_wrong_prediction_data(symbol, strategy, input_size, window=best_window)
                if wrong_data:
                    try:
                        xb_all, yb_all = [], []
                        for xb, yb in wrong_data:
                            if xb.shape[1:] == (best_window, input_size):
                                xb_all.append(xb)
                                yb_all.append(yb)
                        if len(xb_all) >= 2:
                            xb_all = torch.stack(xb_all)
                            yb_all = torch.tensor(yb_all, dtype=torch.float32)
                            for i in range(0, len(xb_all), batch_size):
                                xb = xb_all[i:i + batch_size]
                                yb = yb_all[i:i + batch_size]
                                pred, _ = model(xb)
                                if pred is not None:
                                    loss = criterion(pred, yb)
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                    except Exception as e:
                        print(f"[오답 학습 실패] {symbol}-{strategy} → {e}")

            for epoch in range(epochs):
                for xb, yb in train_loader:
                    pred, _ = model(xb)
                    if pred is None:
                        continue
                    loss = criterion(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        model.eval()
        try:
            with torch.no_grad():
                out, _ = model(val_X_tensor)
                y_prob = out.squeeze().numpy()
                if len(y_prob.shape) == 0:
                    y_prob = np.array([y_prob])
                y_pred = (y_prob > 0.5).astype(int)
                y_true = val_y_tensor.numpy()
                acc = float(accuracy_score(y_true, y_pred))
                f1 = float(f1_score(y_true, y_pred))
                logloss = float(log_loss(y_true, y_prob, labels=[0, 1]))
                conf_score = np.mean(np.abs(y_prob - 0.5)) * 2
                final_score = acc * (1 + f1) * conf_score
                logger.log_training_result(symbol, strategy, model_type, acc, f1, logloss)
                scores[model_type] = final_score
                models[model_type] = model
                metrics[model_type] = (acc, f1, logloss)
        except Exception as e:
            print(f"[평가 오류] {symbol}-{strategy}-{model_type} → {e}")

    if scores:
        best_model_type = max(scores, key=scores.get)
        best_model_obj = models[best_model_type]
        best_acc, best_f1, best_loss = metrics[best_model_type]
        model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{best_model_type}.pt")
        torch.save(best_model_obj.state_dict(), model_path)
        print(f"✅ Best 모델 저장됨: {model_path} (score: {scores[best_model_type]:.4f})")
        save_model_metadata(symbol, strategy, best_model_type, best_acc, best_f1, best_loss)
        importances = compute_feature_importance(best_model_obj, val_X_tensor, val_y_tensor, list(df_feat.columns))
        save_feature_importance(importances, symbol, strategy, best_model_type)
    else:
        print(f"❗ 모델 저장 실패: {symbol}-{strategy} 모든 모델 평가 실패")

def train_all_models():
    for strategy in ["단기", "중기", "장기"]:
        for symbol in SYMBOLS:
            try:
                train_one_model(symbol, strategy)
            except Exception as e:
                print(f"[오류] 전체 학습 실패: {symbol}-{strategy} → {e}")

def train_model_loop(strategy):
    for symbol in SYMBOLS:
        try:
            train_one_model(symbol, strategy)
        except Exception as e:
            print(f"[학습 루프 실패] {symbol}-{strategy} → {e}")

train_model = train_all_models  # ✅ app.py 연동용
