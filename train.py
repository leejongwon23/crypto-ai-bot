import os, json, torch, torch.nn as nn, numpy as np, datetime, pytz, sys, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from data.utils import SYMBOLS, get_kline_by_strategy, compute_features, create_dataset
from model.base_model import get_model
from model_weight_loader import get_model_weight
from feature_importance import compute_feature_importance, save_feature_importance
from wrong_data_loader import load_training_prediction_data
from failure_db import load_existing_failure_hashes
from logger import log_training_result, strategy_stats, load_failure_count
from window_optimizer import find_best_window
import hashlib
from collections import Counter
import sqlite3
from config import NUM_CLASSES


DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
os.makedirs(MODEL_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
STRATEGY_WRONG_REP = {"단기": 4, "중기": 6, "장기": 8}

def get_feature_hash_from_tensor(x):
    if x.ndim != 2 or x.shape[0] == 0:
        return "invalid"
    last = x[-1].tolist()
    rounded = [round(float(val), 2) for val in last]
    return hashlib.sha1(",".join(map(str, rounded)).encode()).hexdigest()

def get_frequent_failures(min_count=5):
    counter = Counter()
    try:
        with sqlite3.connect("/persistent/logs/failure_patterns.db") as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            for row in rows:
                counter[row[0]] += 1
    except:
        return set()
    return {h for h, cnt in counter.items() if cnt >= min_count}

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
    path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"🗘 저장됨: {path}"); sys.stdout.flush()

def train_one_model(symbol, strategy, max_epochs=20):
    import os, gc
    import numpy as np
    import pandas as pd
    import torch
    from collections import Counter
    from model.base_model import get_model
    from feature_importance import compute_feature_importance, save_feature_importance
    from failure_db import load_existing_failure_hashes
    from focal_loss import FocalLoss
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import TensorDataset, DataLoader
    from config import NUM_CLASSES
    from wrong_data_loader import load_training_prediction_data
    from logger import log_training_result, get_feature_hash_from_tensor
    from window_optimizer import find_best_window
    from data.utils import get_kline_by_strategy, compute_features, create_dataset

    print(f"▶ 학습 시작: {symbol}-{strategy}")

    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print("⛔ 중단: get_kline_by_strategy() → 데이터 없음")
            return

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or len(df_feat) < 30 or df_feat.isnull().any().any():
            print("⛔ 중단: compute_features 결과 부족 또는 NaN")
            return

        if "timestamp" not in df_feat.columns:
            df_feat["timestamp"] = df_feat.get("datetime", pd.Timestamp.now())
        df_feat = df_feat.dropna().reset_index(drop=True)
        features = df_feat.to_dict(orient="records")

        window = find_best_window(symbol, strategy)
        if not isinstance(window, int) or window <= 0:
            print(f"⛔ 중단: find_best_window 실패 → {window}")
            return

        X_raw, y_raw = create_dataset(features, window=window, strategy=strategy)
        if X_raw is None or y_raw is None or len(X_raw) < 5:
            print("⛔ 중단: 학습 데이터 생성 실패")
            return

        y_raw = np.array(y_raw)
        X_raw = np.array(X_raw)
        mask = (y_raw >= 0) & (y_raw < NUM_CLASSES)
        y_raw = y_raw[mask]
        X_raw = X_raw[mask]
        if len(X_raw) < 5:
            print(f"⛔ 중단: 유효 학습 샘플 부족 ({len(X_raw)})")
            return

        observed = Counter(int(c) for c in y_raw if c >= 0)
        if len(observed) < 2:
            print(f"⛔ 중단: 클래스 다양성 부족 ({len(observed)}종)")
            return

        input_size = X_raw.shape[2]
        val_len = max(5, int(len(X_raw) * 0.2))
        X_bal, y_bal = balance_classes(X_raw[:-val_len], y_raw[:-val_len], min_samples=20, target_classes=range(NUM_CLASSES))
        X_train, y_train = X_bal, y_bal
        X_val, y_val = X_raw[-val_len:], y_raw[-val_len:]

        # 실패 학습 데이터 반영
        failure_hashes = load_existing_failure_hashes()
        wrong_data = load_training_prediction_data(symbol, strategy, input_size, window)
        wrong_filtered, used_hashes = [], set()
        for xb, yb in wrong_data:
            if not isinstance(xb, np.ndarray) or xb.shape != (window, input_size):
                continue
            if not isinstance(yb, int) or not (0 <= yb < NUM_CLASSES):
                continue
            feature_hash = get_feature_hash_from_tensor(torch.tensor(xb))
            if feature_hash in used_hashes or feature_hash in failure_hashes:
                continue
            used_hashes.add(feature_hash)
            wrong_filtered.append((xb, yb))

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size=input_size, output_size=NUM_CLASSES).train()
            model_path = f"/persistent/models/{symbol}_{strategy}_{model_type}.pt"
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location="cpu"))
                    print(f"🔁 이어 학습: {model_path}")
                except:
                    print(f"[로드 실패] {model_path} → 새로 학습")

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossfn = FocalLoss(gamma=2)

            def train_failures(data, repeat=4):
                ds = TensorDataset(torch.tensor([x for x, _ in data], dtype=torch.float32),
                                   torch.tensor([y for _, y in data], dtype=torch.long))
                loader = DataLoader(ds, batch_size=16, shuffle=True)
                for _ in range(repeat):
                    for xb, yb in loader:
                        model.train()
                        logits = model(xb)
                        loss = lossfn(logits, yb)
                        if torch.isfinite(loss):
                            optimizer.zero_grad(); loss.backward(); optimizer.step()

            if wrong_filtered:
                train_failures(wrong_filtered)

            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            for _ in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    logits = model(xb)
                    loss = lossfn(logits, yb)
                    if torch.isfinite(loss):
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_val, dtype=torch.float32)
                yb = torch.tensor(y_val, dtype=torch.long)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).numpy()
                acc = accuracy_score(y_val, preds)
                f1 = f1_score(y_val, preds, average="macro")
                val_loss = lossfn(logits, yb).item()
                print(f"[검증 성능] acc={acc:.4f}, f1={f1:.4f}, loss={val_loss:.4f}")

            if acc >= 1.0 and len(set(y_val)) <= 2:
                log_training_result(symbol, strategy, f"오버핏({model_type})", acc, f1, val_loss)
                continue
            if f1 > 1.0 or val_loss > 1.5 or acc < 0.3:
                log_training_result(symbol, strategy, f"비정상({model_type})", acc, f1, val_loss)
                continue

            torch.save(model.state_dict(), model_path)
            save_model_metadata(symbol, strategy, model_type, acc, f1, val_loss)
            log_training_result(symbol, strategy, model_type, acc, f1, val_loss)

            try:
                imps = compute_feature_importance(model, xb, yb, list(df_feat.drop(columns=["timestamp"]).columns))
                save_feature_importance(imps, symbol, strategy, model_type)
            except:
                pass

            del model, xb, yb, logits
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"[오류] {symbol}-{strategy} → {e}")
        try:
            log_training_result(symbol, strategy, f"실패({str(e)})", 0.0, 0.0, 0.0)
        except:
            print("⚠️ 로그 기록 실패")


def train_all_models():
    for strat in ["단기", "중기", "장기"]:
        for sym in SYMBOLS:
            try:
                train_one_model(sym, strat)
            except Exception as e:
                print(f"[전체 학습 오류] {sym}-{strat} → {e}")
training_in_progress = {}

def train_model_loop(strategy):
    global training_in_progress
    if training_in_progress.get(strategy, False):
        print(f"⚠️ 이미 실행 중: {strategy} 학습 중복 방지")
        return
    training_in_progress[strategy] = True

    try:
        for symbol in SYMBOLS:
            try:
                train_one_model(symbol, strategy)
            except Exception as e:
                print(f"[학습 실패] {symbol}-{strategy} → {e}")
    finally:
        training_in_progress[strategy] = False
        
def balance_classes(X, y, min_samples=20, target_classes=None):
    from collections import Counter
    import random
    import numpy as np

    if target_classes is None:
        target_classes = range(NUM_CLASSES)  # ✅ NUM_CLASSES = 21 기준으로 적용

    class_counts = Counter(y)
    X_balanced, y_balanced = list(X), list(y)

    for cls in target_classes:
        count = class_counts.get(cls, 0)
        if count == 0:
            continue  # 아예 없는 클래스는 건너뜀
        if count >= min_samples:
            continue  # 충분히 많으면 건너뜀

        existing = [(x, y_val) for x, y_val in zip(X, y) if y_val == cls]
        while class_counts[cls] < min_samples and existing:
            x_dup, y_dup = random.choice(existing)
            X_balanced.append(x_dup)
            y_balanced.append(y_dup)
            class_counts[cls] += 1

    return np.array(X_balanced), np.array(y_balanced)
