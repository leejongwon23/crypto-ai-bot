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

DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
os.makedirs(MODEL_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
NUM_CLASSES = 18
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
    from logger import get_fine_tune_targets, get_recent_predicted_classes
    from focal_loss import FocalLoss
    print(f"▶ 학습 시작: {symbol}-{strategy}")

    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print("⏭ 데이터 없음"); return

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or len(df_feat) < 30:
            print("⏭ 피처 부족"); return

        if "timestamp" not in df_feat.columns:
            df_feat["timestamp"] = df_feat.get("datetime", pd.Timestamp.now())
        df_feat = df_feat.dropna()

        features = df_feat.to_dict(orient="records")
        window = find_best_window(symbol, strategy)
        if not isinstance(window, int) or window <= 0:
            print(f"[스킵] {symbol}-{strategy} → find_best_window 실패 또는 무효값")
            return

        result = create_dataset(features, window=window, strategy=strategy)
        if not result or not isinstance(result, (list, tuple)) or len(result) != 2:
            print(f"[스킵] {symbol}-{strategy} → create_dataset 결과 없음")
            return

        X_raw, y_raw = result
        if X_raw is None or y_raw is None or len(X_raw) < 5:
            print("⏭ 학습용 시퀀스 부족"); return

        input_size = X_raw.shape[2]
        num_classes = int(np.max(y_raw)) + 1
        val_len = int(len(X_raw) * 0.2)
        if val_len == 0:
            print("⏭ 검증 데이터 부족"); return

        # ✅ 클래스 균형 보정 + 미등장 클래스 강제 포함
        target_classes = set(range(NUM_CLASSES))
        observed_classes = set(int(c) for c in np.unique(y_raw))
        missing_classes = list(target_classes - observed_classes)

        X_bal, y_bal = balance_classes(X_raw[:-val_len], y_raw[:-val_len], min_samples=20, target_classes=range(num_classes))
        for cls in missing_classes:
            # 강제 dummy 삽입 (X 중 1개 복사 + 라벨만 변경)
            x_dummy = X_bal[0].copy()
            X_bal = np.vstack([X_bal, [x_dummy]])
            y_bal = np.append(y_bal, cls)

        X_train, y_train = X_bal, y_bal
        X_val, y_val = X_raw[-val_len:], y_raw[-val_len:]

        failure_hashes = load_existing_failure_hashes()
        frequent_failures = get_frequent_failures(min_count=5)
        wrong_data = load_training_prediction_data(symbol, strategy, input_size, window, source_type="wrong")

        from logger import get_feature_hash_from_tensor
        wrong_filtered = []
        used_hashes = set()
        for s in wrong_data:
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                xb, yb = s[:2]
                if not isinstance(xb, np.ndarray) or xb.shape != (window, input_size): continue
                if not isinstance(yb, (int, np.integer)) or not (0 <= yb < NUM_CLASSES): continue
                feature_hash = get_feature_hash_from_tensor(torch.tensor(xb))
                if feature_hash in used_hashes or feature_hash in failure_hashes or feature_hash in frequent_failures:
                    continue
                used_hashes.add(feature_hash)
                wrong_filtered.append((xb, yb))

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size=input_size, output_size=NUM_CLASSES).train()
            model_path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.pt"
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    print(f"🔁 이어 학습: {model_path}")
                except:
                    print(f"[로드 실패] {model_path} → 새로 학습")

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossfn = FocalLoss(gamma=2, weight=None, reduction="mean")

            def train_failures(batch_data, repeat=6):
                ds = TensorDataset(torch.tensor([x for x, _ in batch_data], dtype=torch.float32),
                                   torch.tensor([y for _, y in batch_data], dtype=torch.long))
                loader = DataLoader(ds, batch_size=16, shuffle=True)
                for _ in range(repeat):
                    for xb, yb in loader:
                        model.train()
                        logits = model(xb)
                        loss = lossfn(logits, yb)
                        if not torch.isfinite(loss): continue
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

            train_failures([(x, y) for x, y in wrong_filtered if y >= 10], repeat=6)
            train_failures([(x, y) for x, y in wrong_filtered if y < 10], repeat=2)

            try:
                target_class_set = set()
                recent_pred_classes = get_recent_predicted_classes(strategy, recent_days=3)
                fine_tune_targets = get_fine_tune_targets()
                if recent_pred_classes:
                    target_class_set.update([(strategy, c) for c in recent_pred_classes])
                for _, row in fine_tune_targets.iterrows():
                    target_class_set.add((row["strategy"], row["class"]))
                train_failures([(x, y) for x, y in wrong_filtered if (strategy, y) in target_class_set], repeat=6)
            except:
                print("⚠️ fine-tune 대상 분석 실패 → 전체 학습 유지")

            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            for _ in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    logits = model(xb)
                    loss = lossfn(logits, yb)
                    if not torch.isfinite(loss): break
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

            if acc >= 1.0 and len(set(y_val)) <= 2:
                print(f"⚠️ 오버핏 감지 → 저장 중단")
                log_training_result(symbol, strategy, f"오버핏({model_type})", acc, f1, val_loss)
                continue
            if f1 > 1.0 or val_loss > 1.5 or acc < 0.3:
                print(f"⚠️ 비정상 결과 감지 → 저장 중단 (acc={acc:.2f}, f1={f1:.2f}, loss={val_loss:.2f})")
                log_training_result(symbol, strategy, f"비정상({model_type})", acc, f1, val_loss)
                continue

            torch.save(model.state_dict(), model_path)
            save_model_metadata(symbol, strategy, model_type, acc, f1, val_loss)
            log_training_result(symbol, strategy, model_type, acc, f1, val_loss)

            try:
                imps = compute_feature_importance(model, xb, yb, list(df_feat.drop(columns=["timestamp"]).columns))
                save_feature_importance(imps, symbol, strategy, model_type)
            except:
                print("⚠️ 중요도 저장 실패 (무시됨)")

    except Exception as e:
        print(f"[오류] {symbol}-{strategy} → {e}")
        try:
            log_training_result(symbol, strategy, f"실패({str(e)})", 0.0, 0.0, 0.0)
        except:
            print("⚠️ 로그 기록 실패")

            def train_failures(batch_data, repeat=6):
                ds = TensorDataset(torch.tensor([x for x, _ in batch_data], dtype=torch.float32),
                                   torch.tensor([y for _, y in batch_data], dtype=torch.long))
                loader = DataLoader(ds, batch_size=16, shuffle=True)
                for _ in range(repeat):
                    for xb, yb in loader:
                        model.train()
                        logits = model(xb)
                        loss = lossfn(logits, yb)
                        if not torch.isfinite(loss): continue
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

            train_failures([(x, y) for x, y in wrong_filtered if y >= 10], repeat=6)
            train_failures([(x, y) for x, y in wrong_filtered if y < 10], repeat=2)

            try:
                target_class_set = set()
                recent_pred_classes = get_recent_predicted_classes(strategy, recent_days=3)
                fine_tune_targets = get_fine_tune_targets()
                if recent_pred_classes:
                    target_class_set.update([(strategy, c) for c in recent_pred_classes])
                for _, row in fine_tune_targets.iterrows():
                    target_class_set.add((row["strategy"], row["class"]))
                train_failures([(x, y) for x, y in wrong_filtered if (strategy, y) in target_class_set], repeat=6)
            except:
                print("⚠️ fine-tune 대상 분석 실패 → 전체 학습 유지")

            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            for _ in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    logits = model(xb)
                    loss = lossfn(logits, yb)
                    if not torch.isfinite(loss): break
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

            if acc >= 1.0 and len(set(y_val)) <= 2:
                print(f"⚠️ 오버핏 감지 → 저장 중단")
                log_training_result(symbol, strategy, f"오버핏({model_type})", acc, f1, val_loss)
                continue
            if f1 > 1.0 or val_loss > 1.5 or acc < 0.3:
                print(f"⚠️ 비정상 결과 감지 → 저장 중단 (acc={acc:.2f}, f1={f1:.2f}, loss={val_loss:.2f})")
                log_training_result(symbol, strategy, f"비정상({model_type})", acc, f1, val_loss)
                continue

            torch.save(model.state_dict(), model_path)
            save_model_metadata(symbol, strategy, model_type, acc, f1, val_loss)
            log_training_result(symbol, strategy, model_type, acc, f1, val_loss)

            try:
                imps = compute_feature_importance(model, xb, yb, list(df_feat.drop(columns=["timestamp"]).columns))
                save_feature_importance(imps, symbol, strategy, model_type)
            except:
                print("⚠️ 중요도 저장 실패 (무시됨)")

    except Exception as e:
        print(f"[오류] {symbol}-{strategy} → {e}")
        try:
            log_training_result(symbol, strategy, f"실패({str(e)})", 0.0, 0.0, 0.0)
        except:
            print("⚠️ 로그 기록 실패")


def train_all_models():
    for strat in ["단기", "중기", "장기"]:
        for sym in SYMBOLS:
            try: train_one_model(sym, strat)
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


def balance_classes(X, y, min_samples=20, target_classes=range(18)):
    from collections import Counter
    import random
    import numpy as np

    class_counts = Counter(y)
    X_balanced, y_balanced = list(X), list(y)

    for cls in target_classes:
        count = class_counts.get(cls, 0)
        if count == 0:
            continue  # 아예 없는 클래스는 건너뜀
        if count >= min_samples:
            continue  # 충분히 많으면 건너뜀

        existing = [(x, y_val) for x, y_val in zip(X, y) if y_val == cls]
        while class_counts[cls] < min_samples:
            x_dup, y_dup = random.choice(existing)
            X_balanced.append(x_dup)
            y_balanced.append(y_dup)
            class_counts[cls] += 1

    return np.array(X_balanced), np.array(y_balanced)


