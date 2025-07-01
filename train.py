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
import time

training_in_progress = {"단기": False, "중기": False, "장기": False}


DEVICE = torch.device("cpu")
MODEL_DIR = "/persistent/models"
os.makedirs(MODEL_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
STRATEGY_WRONG_REP = {"단기": 4, "중기": 6, "장기": 8}

def get_feature_hash_from_tensor(x, use_full=False, precision=3):
    """
    ✅ [설명]
    - 마지막 timestep 또는 전체 feature를 반올림 후 sha1 해시값으로 변환
    """
    import hashlib
    if x.ndim != 2 or x.shape[0] == 0:
        return "invalid"
    try:
        flat = x.flatten() if use_full else x[-1]
        rounded = [round(float(val), precision) for val in flat]
        return hashlib.sha1(",".join(map(str, rounded)).encode()).hexdigest()
    except Exception as e:
        # ⚠️ 로그 간소화: 오류시 간단 출력
        print(f"[get_feature_hash_from_tensor 오류] {e}")
        return "invalid"

def get_frequent_failures(min_count=5):
    """
    ✅ [설명] failure_patterns.db에서 동일 실패가 min_count 이상이면 반환
    """
    counter = Counter()
    try:
        with sqlite3.connect("/persistent/logs/failure_patterns.db") as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            for row in rows:
                counter[row[0]] += 1
    except:
        pass
    return {h for h, cnt in counter.items() if cnt >= min_count}


def save_model_metadata(symbol, strategy, model_type, acc, f1, loss, input_size=None, class_counts=None):
    """
    ✅ [설명] 모델 메타정보를 json으로 저장
    """
    meta = {
        "symbol": symbol, "strategy": strategy, "model": model_type or "unknown",
        "input_size": int(input_size) if input_size else 11,
        "accuracy": float(round(acc, 4)), "f1_score": float(round(f1, 4)),
        "loss": float(round(loss, 6)),
        "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
    }
    if class_counts:
        meta["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}
    path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.meta.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        # ✅ 로그 간소화
        print(f"[메타저장] {model_type} ({symbol}-{strategy}) acc={acc:.4f}")
    except Exception as e:
        print(f"[ERROR] meta 저장 실패: {e}")


def train_one_model(symbol, strategy, max_epochs=20):
    """
    ✅ [설명] 한 심볼-전략 모델 학습 수행
    """
    import os, gc
    from focal_loss import FocalLoss
    print(f"▶ 학습 시작: {symbol}-{strategy}")

    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print("⛔ 중단: 시세 데이터 없음")
            return

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
            print("⛔ 중단: 피처 생성 실패 또는 NaN")
            return

        window = find_best_window(symbol, strategy)
        if not isinstance(window, int) or window <= 0:
            print(f"⛔ 중단: find_best_window 실패")
            return

        X_raw, y_raw = create_dataset(df_feat.to_dict(orient="records"), window=window, strategy=strategy)
        if X_raw is None or y_raw is None or len(X_raw) < 5:
            print("⛔ 중단: 학습 데이터 부족")
            return

        y_raw, X_raw = np.array(y_raw), np.array(X_raw, dtype=np.float32)
        mask = (y_raw >= 0) & (y_raw < NUM_CLASSES)
        X_raw, y_raw = X_raw[mask], y_raw[mask]
        if len(X_raw) < 5:
            print("⛔ 중단: 유효 샘플 부족")
            return

        input_size = X_raw.shape[2]
        val_len = max(5, int(len(X_raw) * 0.2))
        X_train, y_train, X_val, y_val = X_raw[:-val_len], y_raw[:-val_len], X_raw[-val_len:], y_raw[-val_len:]

        wrong_data = load_training_prediction_data(symbol, strategy, input_size, window)
        wrong_ds = TensorDataset(torch.tensor([x for x, _ in wrong_data], dtype=torch.float32),
                                 torch.tensor([y for _, y in wrong_data], dtype=torch.long)) if wrong_data else None

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size=input_size, output_size=NUM_CLASSES).to(DEVICE).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossfn = FocalLoss(gamma=2)

            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

            # 🔁 실패 집중 학습
            if wrong_ds:
                wrong_loader = DataLoader(wrong_ds, batch_size=16, shuffle=True, num_workers=2)
                for _ in range(3):
                    for xb, yb in wrong_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        logits = model(xb)
                        loss = lossfn(logits, yb)
                        if torch.isfinite(loss):
                            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # 🔁 기본 학습
            for _ in range(max_epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = lossfn(logits, yb)
                    if torch.isfinite(loss):
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

            # ✅ 검증
            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                yb = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                acc = accuracy_score(y_val, preds)
                f1 = f1_score(y_val, preds, average="macro")
                val_loss = lossfn(logits, yb).item()
                print(f"[검증] {model_type} acc={acc:.4f}, f1={f1:.4f}")

            # ✅ 메타 저장
            log_training_result(symbol, strategy, model_type, acc, f1, val_loss)
            torch.save(model.state_dict(), f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.pt")
            save_model_metadata(symbol, strategy, model_type, acc, f1, val_loss,
                                input_size=input_size, class_counts=Counter(y_train))

            # ✅ feature importance 저장 (오류 무시)
            try:
                imps = compute_feature_importance(model, xb, yb, list(df_feat.drop(columns=["timestamp"]).columns))
                save_feature_importance(imps, symbol, strategy, model_type)
            except:
                pass

            del model, xb, yb, logits
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"[ERROR] {symbol}-{strategy}: {e}")
        log_training_result(symbol, strategy, f"실패({str(e)})", 0.0, 0.0, 0.0)


def balance_classes(X, y, min_samples=20, target_classes=None):
    """
    ✅ [설명] 클래스 불균형 보완
    - 없는 클래스는 noise 샘플 생성
    """
    from collections import Counter
    import random
    import numpy as np

    if target_classes is None:
        target_classes = range(NUM_CLASSES)

    class_counts = Counter(y)
    max_count = max(class_counts.values()) if class_counts else 0
    X_balanced, y_balanced = list(X), list(y)

    # ✅ 데이터 평균, 표준편차 계산
    if len(X) > 0:
        all_data = np.concatenate(X, axis=0)
        data_mean = np.mean(all_data, axis=0)
        data_std = np.std(all_data, axis=0) + 1e-6
    else:
        data_mean = 0.0
        data_std = 1.0

    for cls in target_classes:
        count = class_counts.get(cls, 0)

        # ✅ 없는 클래스는 noise 생성
        if count == 0:
            sample_shape = X[0].shape if len(X) > 0 else (10, 10)
            noise_sample = np.random.normal(loc=data_mean, scale=data_std, size=sample_shape).astype(np.float32)
            X_balanced.append(noise_sample)
            y_balanced.append(cls)
            class_counts[cls] = 1
            print(f"[생성] zero 클래스 {cls} noise 샘플 추가")

        existing = [(x, y_val) for x, y_val in zip(X, y) if y_val == cls]
        while class_counts[cls] < max(min_samples, int(max_count * 0.8)) and existing:
            x_dup, y_dup = random.choice(existing)
            x_aug = x_dup + np.random.normal(loc=0.0, scale=0.01, size=x_dup.shape).astype(np.float32)
            X_balanced.append(x_aug)
            y_balanced.append(y_dup)
            class_counts[cls] += 1

    # ✅ 최종 클래스 분포 간략 출력
    summary = {cls: class_counts.get(cls, 0) for cls in target_classes}
    print(f"[클래스 복제 완료] 분포: {summary}")

    return np.array(X_balanced), np.array(y_balanced)

def train_all_models():
    """
    ✅ [설명] SYMBOLS 전체에 대해 단기, 중기, 장기 학습 수행
    """
    global training_in_progress
    from telegram_bot import send_message
    strategies = ["단기", "중기", "장기"]

    for strategy in strategies:
        if training_in_progress.get(strategy, False):
            print(f"⚠️ 중복 실행 방지: {strategy}")
            continue

        print(f"🚀 {strategy} 학습 시작")
        training_in_progress[strategy] = True

        try:
            for symbol in SYMBOLS:
                try:
                    train_one_model(symbol, strategy)
                except Exception as e:
                    print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")
        except Exception as e:
            print(f"[치명 오류] {strategy} 학습 중단: {e}")
        finally:
            training_in_progress[strategy] = False
            print(f"✅ {strategy} 학습 완료")

        time.sleep(5)

    send_message("✅ 전체 학습 완료. 예측을 실행해주세요.")


def train_models(symbol_list):
    """
    ✅ [설명] 특정 symbol_list에 대해 단기, 중기, 장기 학습 수행
    - meta 보정 후 예측까지 자동 실행
    """
    global training_in_progress
    from telegram_bot import send_message
    from predict_test import main as run_prediction
    import maintenance_fix_meta

    strategies = ["단기", "중기", "장기"]

    for strategy in strategies:
        if training_in_progress.get(strategy, False):
            print(f"⚠️ 중복 실행 방지: {strategy}")
            continue

        print(f"🚀 {strategy} 학습 시작")
        training_in_progress[strategy] = True

        try:
            for symbol in symbol_list:
                try:
                    train_one_model(symbol, strategy)
                except Exception as e:
                    print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")
        except Exception as e:
            print(f"[치명 오류] {strategy} 전체 학습 중단: {e}")
        finally:
            training_in_progress[strategy] = False
            print(f"✅ {strategy} 학습 완료")

        time.sleep(5)

        try:
            maintenance_fix_meta.fix_all_meta_json()
            print(f"✅ meta 보정 완료: {strategy}")
        except Exception as e:
            print(f"[⚠️ meta 보정 실패] {e}")

        try:
            run_prediction(strategy, symbols=symbol_list)
            print(f"✅ 예측 완료: {strategy}")
        except Exception as e:
            print(f"❌ 예측 실패: {strategy} → {e}")

    send_message("✅ 학습 및 예측 완료")

def train_model_loop(strategy):
    """
    ✅ [설명] 특정 strategy 학습을 무한 루프로 실행
    - training_in_progress 상태 관리 포함
    """
    global training_in_progress
    if training_in_progress.get(strategy, False):
        print(f"⚠️ 중복 실행 방지: {strategy}")
        return

    training_in_progress[strategy] = True
    print(f"🚀 {strategy} 무한 학습 루프 시작")

    try:
        for symbol in SYMBOLS:
            try:
                train_one_model(symbol, strategy)
            except Exception as e:
                print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")
    finally:
        training_in_progress[strategy] = False
        print(f"✅ {strategy} 루프 종료")


def train_symbol_group_loop(delay_minutes=5):
    """
    ✅ [설명] SYMBOL_GROUPS 단위로 전체 그룹 학습 루프 실행
    - 각 그룹 학습 전 cache clear
    - 각 그룹 학습 후 meta 보정, 예측 실행 포함
    """
    import time
    import maintenance_fix_meta
    from data.utils import SYMBOL_GROUPS, _kline_cache, _feature_cache

    group_count = len(SYMBOL_GROUPS)
    print(f"🚀 전체 {group_count}개 그룹 학습 루프 시작")

    while True:
        for idx, group in enumerate(SYMBOL_GROUPS):
            print(f"\n🚀 [그룹 {idx}] 학습 시작 → {group}")

            # ✅ 캐시 clear 추가
            _kline_cache.clear()
            _feature_cache.clear()
            print("[✅ cache cleared] _kline_cache, _feature_cache")

            try:
                train_models(group)

                maintenance_fix_meta.fix_all_meta_json()
                print(f"✅ meta 보정 완료: 그룹 {idx}")

                # ✅ 예측 실행
                for symbol in group:
                    for strategy in ["단기", "중기", "장기"]:
                        try:
                            from recommend import main
                            main(symbol=symbol, strategy=strategy, force=True)
                            print(f"✅ 예측 완료: {symbol}-{strategy}")
                        except Exception as e:
                            print(f"❌ 예측 실패: {symbol}-{strategy} → {e}")

                print(f"🕒 그룹 {idx} 완료 → {delay_minutes}분 대기")
                time.sleep(delay_minutes * 60)

            except Exception as e:
                print(f"❌ 그룹 {idx} 루프 중 오류: {e}")
                continue

