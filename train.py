import os, json, torch, torch.nn as nn, numpy as np, datetime, pytz, sys, pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from data.utils import SYMBOLS, get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from wrong_data_loader import load_training_prediction_data  # ✅ 함수명 수정 완료
from feature_importance import compute_feature_importance, save_feature_importance
import logger
from logger import strategy_stats

DEVICE = torch.device("cpu")
DIR = "/persistent"; MODEL_DIR, LOG_DIR = f"{DIR}/models", f"{DIR}/logs"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def find_best_window(symbol, strategy, window_list=[10, 20, 30, 40]):
    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 10:
            return 20
        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or len(df_feat) < max(window_list) + 1:
            return 20
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_feat.values)
        feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]
        best_score, best_window = -1, window_list[0]
        for window in window_list:
            X, y = create_dataset(feature_dicts, window)
            if len(X) == 0: continue
            input_size = X.shape[2]
            model = get_model("lstm", input_size=input_size)
            model.train()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            val_len = int(len(X_tensor) * 0.2)
            if val_len == 0: continue
            train_X, train_y = X_tensor[:-val_len], y_tensor[:-val_len]
            val_X, val_y = X_tensor[-val_len:], y_tensor[-val_len:]
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            for _ in range(3):
                pred = model(train_X).squeeze()
                loss = criterion(pred, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                pred_val = model(val_X).squeeze().numpy()
                acc = r2_score(val_y.numpy(), pred_val)
                conf = np.mean(np.abs(pred_val))
                score = acc * conf
                if score > best_score:
                    best_score = score
                    best_window = window
    except Exception as e:
        print(f"[find_best_window 오류] {symbol}-{strategy} → {e}")
        return 20
    return best_window

def create_dataset(f, w):
    X, y = [], []
    for i in range(len(f) - w - 1):
        x_seq = f[i:i + w]
        if any(len(r.values()) != len(f[0].values()) for r in x_seq): continue
        c1, c2 = f[i + w - 1]['close'], f[i + w]['close']
        if c1 == 0: continue
        X.append([list(r.values()) for r in x_seq])
        y.append(round((c2 - c1) / c1, 4))
    if not X: return np.array([]), np.array([])
    mlen = max(set(map(len, X)), key=list(X).count)
    filt = [(x, l) for x, l in zip(X, y) if len(x) == mlen]
    if not filt: return np.array([]), np.array([])
    return np.array([x for x, _ in filt]), np.array([l for _, l in filt])

def save_model_metadata(s, t, m, a, f1, l):
    meta = {
        "symbol": s,
        "strategy": t,
        "model": m,
        "accuracy": float(round(a,4)),
        "f1_score": float(round(f1,4)),
        "loss": float(round(l,6)),
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
    }
    path = f"{MODEL_DIR}/{s}_{t}_{m}.meta.json"
    with open(path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"🗘 저장됨: {path}"); sys.stdout.flush()

def train_one_model(sym, strat, input_size=11, batch=32, epochs=10, lr=1e-3, rep=8, rep_wrong=8):
    print(f"[train] 🔄 {sym}-{strat} 시작"); sys.stdout.flush()
    try:
        win = find_best_window(sym, strat)
        df = get_kline_by_strategy(sym, strat)
        if df is None or len(df) < win + 10: raise ValueError("데이터 부족")
        df_feat = compute_features(sym, df, strat)
        if df_feat is None or len(df_feat) < win + 1: raise ValueError("feature 부족")
        feat = MinMaxScaler().fit_transform(df_feat.values)
        X_raw, y_raw = create_dataset([dict(zip(df_feat.columns, r)) for r in feat], win)
        if len(X_raw) < 2: raise ValueError("유효 시퀀스 부족")
        input_size = X_raw.shape[2]
        val_len = int(len(X_raw) * 0.2)
        if val_len == 0: raise ValueError("검증셋 부족")
        val_X = torch.tensor(X_raw[-val_len:], dtype=torch.float32)
        val_y = torch.tensor(y_raw[-val_len:], dtype=torch.float32).view(-1)
        dataset = TensorDataset(torch.tensor(X_raw, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32))
        train_set, _ = random_split(dataset, [len(dataset)-val_len, val_len])

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size); model.train()
            model_path = f"{MODEL_DIR}/{sym}_{strat}_{model_type}.pt"
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    print(f"🔁 이어 학습: {model_path}"); sys.stdout.flush()
                except Exception as e:
                    print(f"[로드 실패 → 새로 학습] {model_path} → {e}"); sys.stdout.flush()

            optim, lossfn = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()
            loader = DataLoader(train_set, batch_size=batch, shuffle=True)

            try:
                with torch.no_grad():
                    before_pred = model(val_X).squeeze(-1).numpy()
                    acc_before = r2_score(val_y.numpy(), before_pred)
            except:
                acc_before = ""

            for _ in range(epochs):
                for _ in range(rep_wrong):
                    wrong_data = load_training_prediction_data(sym, strat, input_size, win, source_type="both")  # ✅ source_type 포함
                    if not wrong_data: continue
                    xb_all, yb_all = zip(*[(xb, yb) for xb, yb in wrong_data
                                           if xb.shape[1:] == (win, input_size) and np.isfinite(yb) and abs(yb) < 2]) if wrong_data else ([],[])
                    if len(xb_all) >= 2:
                        xb = torch.stack(xb_all)
                        yb = torch.tensor(yb_all, dtype=torch.float32).view(-1)
                        for i in range(0, len(xb), batch):
                            rate = model(xb[i:i+batch]).squeeze(-1)
                            loss = lossfn(rate, yb[i:i+batch])
                            optim.zero_grad(); loss.backward(); optim.step()

                for xb, yb in loader:
                    rate = model(xb).squeeze(-1)
                    loss = lossfn(rate, yb)
                    optim.zero_grad(); loss.backward(); optim.step()

            model.eval()
            try:
                with torch.no_grad():
                    rate = model(val_X).squeeze(-1).numpy()
                    acc = r2_score(val_y.numpy(), rate)
                    f1 = mean_squared_error(val_y.numpy(), rate)
                    logloss = np.mean(np.square(val_y.numpy() - rate))
                    acc_dir = accuracy_score(val_y.numpy() > 0, rate > 0)
                    logger.log_training_result(sym, strat, model_type, acc, f1, logloss)
                    torch.save(model.state_dict(), model_path)
                    print(f"✅ 저장: {model_path}"); sys.stdout.flush()
                    save_model_metadata(sym, strat, model_type, acc, f1, logloss)
                    imps = compute_feature_importance(model, val_X, val_y, list(df_feat.columns))
                    save_feature_importance(imps, sym, strat, model_type)
            except Exception as e:
                print(f"[평가 오류] {sym}-{strat}-{model_type} → {e}"); sys.stdout.flush()
    except Exception as e:
        print(f"[실패] {sym}-{strat} → {e}"); sys.stdout.flush()
        try:
            logger.log_training_result(sym, strat, f"실패({str(e)})", 0.0, 0.0, 0.0)
        except Exception as log_err:
            print(f"[로그 기록 실패] {sym}-{strat} → {log_err}"); sys.stdout.flush()

def train_all_models():
    for strat in ["단기", "중기", "장기"]:
        for sym in SYMBOLS:
            try: train_one_model(sym, strat)
            except Exception as e:
                print(f"[전체 학습 오류] {sym}-{strat} → {e}"); sys.stdout.flush()

def train_model_loop(strategy):
    for sym in SYMBOLS:
        try: train_one_model(sym, strategy)
        except Exception as e:
            print(f"[단일 학습 오류] {sym}-{strategy} → {e}"); sys.stdout.flush()

train_model = train_all_models
