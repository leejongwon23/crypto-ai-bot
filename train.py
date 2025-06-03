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

# ✅ 자주 실패한 해시를 찾는 함수
def get_frequent_failures(min_count=5):
    from collections import Counter
    import sqlite3
    DB_PATH = "/persistent/logs/failure_patterns.db"
    counter = Counter()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            for row in rows:
                counter[row[0]] += 1
    except:
        return set()
    return {h for h, cnt in counter.items() if cnt >= min_count}

def train_one_model(sym, strat, input_size=11, batch=32, epochs=10, lr=1e-3, rep=8):
    print(f"[train] 🔄 {sym}-{strat} 시작"); sys.stdout.flush()
    try:
        failmap = load_failure_count()
        key = f"{sym}-{strat}"
        fail_count = failmap.get(key, 0)
        rep_wrong = STRATEGY_WRONG_REP.get(strat, 4)
        if fail_count >= 10:
            rep_wrong += 4
        elif fail_count >= 5:
            rep_wrong += 2

        win = find_best_window(sym, strat)
        df = get_kline_by_strategy(sym, strat)
        if df is None or len(df) < win + 10:
            raise ValueError("데이터 부족")

        df_feat = compute_features(sym, df, strat)
        if df_feat is None or len(df_feat) < win + 1:
            raise ValueError("feature 부족")

        scaled = MinMaxScaler().fit_transform(df_feat.drop(columns=["timestamp"]).values)
        feat = []
        for i, row in enumerate(scaled):
            d = dict(zip(df_feat.columns.drop("timestamp"), row))
            d["timestamp"] = df_feat.iloc[i]["timestamp"]
            feat.append(d)

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
        from failure_db import get_frequent_failures, group_failures_by_reason
        frequent_failures = get_frequent_failures(min_count=5)
        top_failure_reasons = [r["reason"] for r in group_failures_by_reason(limit=3)]

        min_wrong_repeat = max(rep_wrong, 3)

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

            for _ in range(min_wrong_repeat):
                wrong_data = load_training_prediction_data(sym, strat, input_size, win, source_type="wrong")
                if not wrong_data: continue

                xb_all, yb_all = [], []
                for sample in wrong_data:
                    if len(sample) == 3:
                        xb, yb, reason = sample
                        if reason not in top_failure_reasons:
                            continue
                    else:
                        xb, yb = sample[:2]
                    if xb.shape[1:] != (win, input_size) or not np.isfinite(yb) or abs(yb) >= 2:
                        continue
                    xb_all.append(xb)
                    yb_all.append(yb)

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
                            if (sym, strat, direction, feature_hash) in failure_hashes or feature_hash in frequent_failures:
                                continue
                            rate = model(xb_j)
                            if isinstance(rate, tuple): rate = rate[0]
                            rate = rate.view_as(yb_j)
                            loss = lossfn(rate, yb_j)
                            optim.zero_grad(); loss.backward(); optim.step()
                            total_train_count += 1

            for _ in range(epochs):
                for xb, yb in loader:
                    rate = model(xb)
                    if isinstance(rate, tuple): rate = rate[0]
                    rate = rate.view_as(yb)
                    loss = lossfn(rate, yb)
                    optim.zero_grad(); loss.backward(); optim.step()
                    total_train_count += 1

            model.eval()
            with torch.no_grad():
                rate = model(val_X)
                if isinstance(rate, tuple): rate = rate[0]
                rate = rate.view_as(val_y)

                acc = r2_score(val_y.numpy(), rate.numpy())

                # ✅ 방향 기준 정확도
                actual_dir = (val_y.numpy() >= 0).astype(int)
                pred_dir = (rate.numpy() >= 0).astype(int)
                dir_acc = accuracy_score(actual_dir, pred_dir)

                f1 = mean_squared_error(val_y.numpy(), rate.numpy())
                logloss = np.mean(np.square(val_y.numpy() - rate.numpy()))

                logger.log_training_result(sym, strat, model_type, dir_acc, f1, logloss)
                torch.save(model.state_dict(), model_path)
                save_model_metadata(sym, strat, model_type, dir_acc, f1, logloss)
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

def train_model(symbol, strategy, max_epochs=30):
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from data.utils import get_kline_by_strategy, compute_features, create_dataset
    from model.base_model import get_model
    from logger import log_training_result, load_failure_count
    import os, numpy as np

    DEVICE = torch.device("cpu")
    MODEL_DIR = "/persistent/models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[학습 시작] {symbol} - {strategy}")

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print("[스킵] 데이터 없음")
        return

    feat_df = compute_features(symbol, df, strategy)
    if feat_df is None or feat_df.empty:
        print("[스킵] 피처 생성 실패")
        return

    feat_df = feat_df.dropna()
    if "timestamp" not in feat_df.columns:
        print("[스킵] timestamp 없음")
        return

    features = feat_df.to_dict(orient="records")
    window = 20

    X, y = create_dataset(features, window=window, strategy=strategy)
    if X.size == 0 or y.size == 0:
        print("[스킵] 학습용 데이터셋 생성 실패")
        return

    X = MinMaxScaler().fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    failure_map = load_failure_count()
    key = f"{symbol}-{strategy}"
    fail_count = failure_map.get(key, 0)

    # ✅ 실패횟수 기반 반복 학습 횟수 조정
    rep = 1
    if fail_count >= 3:
        rep = 2
    if fail_count >= 6:
        rep = 3
    if strategy == "장기":
        rep += 1  # 고위험 전략 우선 보정

    print(f"[학습 반복 횟수] 실패 {fail_count}회 → rep={rep}")

    for model_type in ["lstm", "cnn_lstm", "transformer"]:
        print(f"▶ 모델 유형: {model_type}")
        try:
            model = get_model(model_type=model_type, input_size=X.shape[2]).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            for _ in range(rep):
                for epoch in range(max_epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE))
                    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).to(DEVICE))
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                preds_bin = (preds > 0).astype(int)
                labels_bin = (y_val > 0).astype(int)
                acc = accuracy_score(labels_bin, preds_bin)
                f1 = f1_score(labels_bin, preds_bin)
                loss_val = criterion(torch.tensor(preds), torch.tensor(y_val)).item()

            model_path = os.path.join(MODEL_DIR, f"{symbol}_{strategy}_{model_type}.pt")
            torch.save(model.state_dict(), model_path)

            log_training_result(symbol, strategy, model_type, acc, f1, loss_val)
            print(f"✅ {model_type} 저장 완료: acc={acc:.3f}, f1={f1:.3f}, loss={loss_val:.4f}")

        except Exception as e:
            print(f"[오류] {model_type} 학습 실패: {e}")



