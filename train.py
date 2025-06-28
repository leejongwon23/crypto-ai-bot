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

def save_model_metadata(symbol, strategy, model_type, acc, f1, loss, input_size=None, class_counts=None):
    meta = {
        "symbol": symbol,
        "strategy": strategy,
        "model": model_type,
        "input_size": input_size,
        "accuracy": float(round(acc, 4)),
        "f1_score": float(round(f1, 4)),
        "loss": float(round(loss, 6)),
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
    }

    # ✅ 클래스 카운트를 문자열 키로 저장
    if class_counts:
        meta["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}

    path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"🗘 저장됨: {path}"); sys.stdout.flush()

from logger import get_fine_tune_targets  # 🔁 반드시 포함


def train_one_model(symbol, strategy, max_epochs=20):
    import os, gc
    import numpy as np
    import pandas as pd
    import torch
    import datetime, pytz
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
    MODEL_DIR = "/persistent/models"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print("⛔ 중단: get_kline_by_strategy() → 데이터 없음")
            return

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
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

        if df_feat.shape[0] < window + 1:
            print(f"⛔ 중단: feature 수 부족 → 필요 {window + 1}, 현재 {df_feat.shape[0]}")
            return

        X_raw, y_raw = create_dataset(features, window=window, strategy=strategy)
        if X_raw is None or y_raw is None or len(X_raw) < 5:
            print("⛔ 중단: 학습 데이터 생성 실패")
            return

        y_raw = np.array(y_raw)
        X_raw = np.array(X_raw, dtype=np.float32)
        mask = (y_raw >= 0) & (y_raw < NUM_CLASSES)
        y_raw = y_raw[mask]
        X_raw = X_raw[mask]

        if len(X_raw) < 5:
            print(f"⛔ 중단: 유효 학습 샘플 부족 ({len(X_raw)})")
            return

        if len(set(y_raw)) < 2:
            print(f"⛔ 중단: 클래스 다양성 부족 ({len(set(y_raw))}종)")
            return

        input_size = X_raw.shape[2]
        val_len = max(5, int(len(X_raw) * 0.2))
        X_bal, y_bal = balance_classes(X_raw[:-val_len], y_raw[:-val_len], min_samples=20, target_classes=range(NUM_CLASSES))
        X_train, y_train = X_bal, y_bal
        X_val, y_val = X_raw[-val_len:], y_raw[-val_len:]

        class_counts = Counter(y_train)  # ✅ 복제 반영된 클래스 정보 저장용

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
            model = get_model(model_type, input_size=input_size, output_size=NUM_CLASSES).to(DEVICE).train()
            model_path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.pt"
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    print(f"🔁 이어 학습: {model_path}")
                except:
                    print(f"[로드 실패] {model_path} → 새로 학습")

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossfn = FocalLoss(gamma=2)

            if wrong_filtered:
                ds = TensorDataset(torch.tensor([x for x, _ in wrong_filtered], dtype=torch.float32),
                                   torch.tensor([y for _, y in wrong_filtered], dtype=torch.long))
                loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
                for _ in range(4):
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        model.train()
                        logits = model(xb)
                        loss = lossfn(logits, yb)
                        if torch.isfinite(loss):
                            optimizer.zero_grad(); loss.backward(); optimizer.step()

            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
            for _ in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = lossfn(logits, yb)
                    if torch.isfinite(loss):
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

            from logger import get_fine_tune_targets
            fine_tune_targets = get_fine_tune_targets()
            if not fine_tune_targets.empty:
                targets = fine_tune_targets[fine_tune_targets["strategy"] == strategy]["class"].tolist()
                if any(cls in targets for cls in y_train):
                    print(f"🔁 Fine-Tune 반복 학습 시작")
                    for _ in range(3):
                        for xb, yb in train_loader:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            logits = model(xb)
                            loss = lossfn(logits, yb)
                            if torch.isfinite(loss):
                                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                yb = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                acc = accuracy_score(y_val, preds)
                f1 = f1_score(y_val, preds, average="macro")
                val_loss = lossfn(logits, yb).item()
                print(f"[검증 성능] acc={acc:.4f}, f1={f1:.4f}, loss={val_loss:.4f}")

            log_training_result(symbol, strategy, model_type, acc, f1, val_loss)

            torch.save(model.state_dict(), model_path)
            save_model_metadata(symbol, strategy, model_type, acc, f1, val_loss,
                                input_size=input_size, class_counts=class_counts)  # ✅ 복제 클래스 저장

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

training_in_progress = {
    "단기": False,
    "중기": False,
    "장기": False
}
import time

def train_all_models():
    from telegram_bot import send_message  # ✅ 메시지 전송 함수 가져오기
    strategies = ["단기", "중기", "장기"]

    for strategy in strategies:
        if training_in_progress.get(strategy, False):
            print(f"⚠️ 이미 실행 중: {strategy} 학습 중복 방지"); continue

        print(f"\n🚀 전략 학습 시작: {strategy}")
        training_in_progress[strategy] = True

        try:
            for symbol in SYMBOLS:
                try:
                    print(f"▶ 학습 시작: {symbol}-{strategy}")
                    train_one_model(symbol, strategy)
                except Exception as e:
                    print(f"[오류] {symbol}-{strategy} 학습 실패 → {e}")
        except Exception as e:
            print(f"[치명 오류] {strategy} 전체 학습 중단 → {type(e).__name__}: {e}")
        finally:
            training_in_progress[strategy] = False
            print(f"✅ 전략 학습 완료: {strategy}\n")

        time.sleep(5)  # ✅ 다음 전략 학습 전 5초 대기 → 병렬 진입 방지

    # ✅ 전체 전략 학습이 모두 끝난 후 텔레그램 메시지 전송
    send_message("✅ 전체 학습이 완료되었습니다. 예측을 실행해주세요.")

def train_models(symbol_list):
    from telegram_bot import send_message
    from predict_test import main as run_prediction
    import maintenance_fix_meta  # ✅ meta 보정 모듈 import

    strategies = ["단기", "중기", "장기"]

    for strategy in strategies:
        if training_in_progress.get(strategy, False):
            print(f"⚠️ 이미 실행 중: {strategy} 학습 중복 방지"); continue

        print(f"\n🚀 전략 학습 시작: {strategy}")
        training_in_progress[strategy] = True

        try:
            for symbol in symbol_list:
                try:
                    print(f"▶ 학습 시작: {symbol}-{strategy}")
                    train_one_model(symbol, strategy)
                except Exception as e:
                    print(f"[오류] {symbol}-{strategy} 학습 실패 → {e}")
        except Exception as e:
            print(f"[치명 오류] {strategy} 전체 학습 중단 → {type(e).__name__}: {e}")
        finally:
            training_in_progress[strategy] = False
            print(f"✅ 전략 학습 완료: {strategy}\n")

        time.sleep(5)  # ✅ 다음 전략 학습 전 5초 대기

        # ✅ 각 전략 학습 후 meta 보정 실행
        try:
            maintenance_fix_meta.fix_all_meta_json()
        except Exception as e:
            print(f"[⚠️ meta 보정 실패] {e}")

        # ✅ meta 보정 후 예측 실행
        try:
            run_prediction(strategy, symbols=symbol_list)  # ✅ 학습 완료된 심볼만 예측
        except Exception as e:
            print(f"❌ 예측 실패: {strategy} → {e}")

    send_message("✅ 학습 및 예측 루틴 완료 (해당 심볼 그룹)")


def train_model_loop(strategy):
    if training_in_progress.get(strategy, False):
        print(f"⚠️ 이미 실행 중: {strategy} 학습 중복 방지")
        return

    training_in_progress[strategy] = True
    print(f"📌 상태 진입 → {training_in_progress}")  # ✅ 상태 확인용

    try:
        for symbol in SYMBOLS:
            try:
                print(f"▶ 학습 시작: {symbol}-{strategy}")
                train_one_model(symbol, strategy)  # ✅ 내부에서 이어 학습 구조 반영됨
            except Exception as e:
                print(f"[학습 실패] {symbol}-{strategy} → {e}")
    finally:
        training_in_progress[strategy] = False
        print(f"📌 상태 종료 → {training_in_progress}")  # ✅ 상태 해제 확인용


def balance_classes(X, y, min_samples=20, target_classes=None):
    from collections import Counter
    import random
    import numpy as np

    if target_classes is None:
        target_classes = range(NUM_CLASSES)

    class_counts = Counter(y)
    max_count = max(class_counts.values()) if class_counts else 0
    X_balanced, y_balanced = list(X), list(y)
    original_counts = dict(class_counts)  # 🔍 복제 전 기록

    for cls in target_classes:
        count = class_counts.get(cls, 0)
        if count == 0:
            continue

        existing = [(x, y_val) for x, y_val in zip(X, y) if y_val == cls]
        while class_counts[cls] < max(min_samples, int(max_count * 0.8)) and existing:
            x_dup, y_dup = random.choice(existing)
            X_balanced.append(x_dup)
            y_balanced.append(y_dup)
            class_counts[cls] += 1

    # ✅ 복제 로그 출력
    print("📊 클래스 복제 현황:")
    for cls in target_classes:
        before = original_counts.get(cls, 0)
        after = class_counts.get(cls, 0)
        if after > before:
            print(f"  - 클래스 {cls}: {before}개 → {after}개 (복제됨)")

    return np.array(X_balanced), np.array(y_balanced)
# ✅ train.py 맨 아래에 반드시 포함해야 함

def train_symbol_group_loop(delay_minutes=5):
    import time
    import maintenance_fix_meta  # ✅ meta 보정 import
    from data.utils import SYMBOL_GROUPS
    group_count = len(SYMBOL_GROUPS)
    print(f"[자동 루프] 전체 {group_count}개 그룹 학습 루프 시작됨")

    while True:
        for idx, group in enumerate(SYMBOL_GROUPS):
            try:
                print(f"\n🚀 [그룹 {idx}] 학습 시작 → {group}")
                
                # ✅ 1. 그룹 학습 먼저 실행
                train_models(group)

                # ✅ 2. meta 보정 (학습 후 예측 전)
                try:
                    maintenance_fix_meta.fix_all_meta_json()
                    print(f"✅ meta 보정 완료 (그룹 {idx})")
                except Exception as e:
                    print(f"[⚠️ meta 보정 실패] {e}")

                # ✅ 3. 예측 실행
                print(f"✅ [그룹 {idx}] 학습 + 보정 완료 → 예측 시작")
                for symbol in group:
                    for strategy in ["단기", "중기", "장기"]:
                        try:
                            from recommend import main
                            main(symbol=symbol, strategy=strategy, force=True)
                        except Exception as e:
                            print(f"❌ 예측 실패: {symbol}-{strategy} → {e}")

                print(f"🕒 [그룹 {idx}] 다음 그룹까지 {delay_minutes}분 대기")
                time.sleep(delay_minutes * 60)

            except Exception as e:
                print(f"❌ 그룹 {idx} 루프 중 오류 발생: {e}")
                continue
