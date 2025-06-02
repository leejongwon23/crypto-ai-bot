import os, json, torch, torch.nn as nn, numpy as np, datetime, pytz, sys, pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from data.utils import SYMBOLS, get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from wrong_data_loader import load_training_prediction_data
from feature_importance import compute_feature_importance, save_feature_importance
import logger
from failure_db import load_existing_failure_hashes  # ✅ 새 DB 기반 실패 해시 로더
from logger import strategy_stats
import csv
import hashlib
from data.utils import create_dataset
from window_optimizer import find_best_window
from logger import load_failure_count  # ✅ 실패횟수 로더 추가

STRATEGY_WRONG_REP = {"단기": 4, "중기": 6, "장기": 8}

DEVICE = torch.device("cpu")
DIR = "/persistent"; MODEL_DIR, LOG_DIR = f"{DIR}/models", f"{DIR}/logs"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def get_feature_hash_from_tensor(x):
    if x.ndim != 2 or x.shape[0] == 0:
        return "invalid"
    last = x[-1].tolist()
    # ✅ 4자리 → 2자리 반올림으로 유사 실패군까지 포착
    rounded = [round(float(val), 2) for val in last]
    return hashlib.sha1(",".join(map(str, rounded)).encode()).hexdigest()

    
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
    print(f"🗘저장됨: {path}"); sys.stdout.flush()

def train_one_model(sym, strat, input_size=11, batch=32, epochs=10, lr=1e-3, rep=8):
    print(f"[train] 🔄 {sym}-{strat} 시작"); sys.stdout.flush()
    try:
        # ✅ 실패 횟수 기반 학습 우선순위 조정
        failmap = load_failure_count()
        key = f"{sym}-{strat}"
        fail_count = failmap.get(key, 0)
        rep_wrong = STRATEGY_WRONG_REP.get(strat, 4)
        if fail_count >= 10:
            rep_wrong += 4
        elif fail_count >= 5:
            rep_wrong += 2

        # ① 데이터 불러오기
        win = find_best_window(sym, strat)
        df = get_kline_by_strategy(sym, strat)
        if df is None or len(df) < win + 10:
            raise ValueError("데이터 부족")

        df_feat = compute_features(sym, df, strat)
        if df_feat is None or len(df_feat) < win + 1:
            raise ValueError("feature 부족")

        # ② feature scaling
        scaled = MinMaxScaler().fit_transform(df_feat.drop(columns=["timestamp"]).values)
        feat = []
        for i, row in enumerate(scaled):
            d = dict(zip(df_feat.columns.drop("timestamp"), row))
            d["timestamp"] = df_feat.iloc[i]["timestamp"]
            feat.append(d)

        # ③ dataset 생성
        X_raw, y_raw = create_dataset(feat, win, strat)
        if len(X_raw) < 2:
            raise ValueError("유효 시퀀스 부족")

        input_size = X_raw.shape[2]
        val_len = int(len(X_raw) * 0.2)
        if val_len == 0:
            raise ValueError("검증셋 부족")

        val_X = torch.tensor(X_raw[-val_len:], dtype=torch.float32)
        val_y = torch.tensor(y_raw[-val_len:], dtype=torch.float32).view(-1)
        dataset = TensorDataset(torch.tensor(X_raw, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32))
        train_set, _ = random_split(dataset, [len(dataset)-val_len, val_len])
        failure_hashes = load_existing_failure_hashes()

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size).train()
            model_path = f"{MODEL_DIR}/{sym}_{strat}_{model_type}.pt"
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    print(f"🔁 이어 학습: {model_path}"); sys.stdout.flush()
                except:
                    print(f"[로드 실패] {model_path} → 새로 학습"); sys.stdout.flush()

            optim = torch.optim.Adam(model.parameters(), lr=lr)
            lossfn = nn.MSELoss()
            loader = DataLoader(train_set, batch_size=batch, shuffle=True)

            total_train_count = 0

            # ④ 실패 샘플 반복 학습
            for _ in range(epochs):
                for _ in range(rep_wrong):
                    wrong_data = load_training_prediction_data(sym, strat, input_size, win, source_type="wrong")
                    if not wrong_data: continue
                    xb_all, yb_all = zip(*[(xb, yb) for xb, yb in wrong_data
                                           if xb.shape[1:] == (win, input_size) and np.isfinite(yb) and abs(yb) < 2]) if wrong_data else ([], [])
                    if len(xb_all) >= 2:
                        xb_tensor = torch.stack(xb_all)
                        yb_tensor = torch.tensor(yb_all, dtype=torch.float32).view(-1)
                        for i in range(0, len(xb_tensor), batch):
                            xb = xb_tensor[i:i+batch]
                            yb = yb_tensor[i:i+batch]
                            for j in range(len(xb)):
                                xb_j = xb[j].unsqueeze(0)
                                yb_j = yb[j].unsqueeze(0)
                                if xb_j.shape[1] == 0:
                                    continue
                                feature_hash = get_feature_hash_from_tensor(xb_j[0])
                                direction = "롱" if yb_j.item() >= 0 else "숏"
                                if (sym, strat, direction, feature_hash) in failure_hashes:
                                    continue
                                rate = model(xb_j)
                                if isinstance(rate, tuple): rate = rate[0]
                                rate = rate.view_as(yb_j)
                                loss = lossfn(rate, yb_j)
                                optim.zero_grad(); loss.backward(); optim.step()
                                total_train_count += 1

            # ⑤ 일반 학습 반복
            for xb, yb in loader:
                rate = model(xb)
                if isinstance(rate, tuple): rate = rate[0]
                rate = rate.view_as(yb)
                loss = lossfn(rate, yb)
                optim.zero_grad(); loss.backward(); optim.step()
                total_train_count += 1

            # ⑥ 학습 평가 및 저장
            model.eval()
            with torch.no_grad():
                rate = model(val_X)
                if isinstance(rate, tuple): rate = rate[0]
                rate = rate.view_as(val_y)
                acc = r2_score(val_y.numpy(), rate.numpy())
                f1 = mean_squared_error(val_y.numpy(), rate.numpy())
                logloss = np.mean(np.square(val_y.numpy() - rate.numpy()))
                logger.log_training_result(sym, strat, model_type, acc, f1, logloss)
                torch.save(model.state_dict(), model_path)
                save_model_metadata(sym, strat, model_type, acc, f1, logloss)
                imps = compute_feature_importance(model, val_X, val_y, list(df_feat.columns))
                save_feature_importance(imps, sym, strat, model_type)

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

def train_model(symbol, strategy):
    train_one_model(symbol, strategy)


