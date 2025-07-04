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


def save_model_metadata(symbol, strategy, model_type, acc, f1, loss, input_size=None, class_counts=None, used_feature_columns=None):
    """
    ✅ [설명] 모델 메타정보를 json으로 저장 (used_feature_columns 포함)
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

    # ✅ 사용된 feature 컬럼 저장
    if used_feature_columns:
        meta["used_feature_columns"] = used_feature_columns

    path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.meta.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[메타저장] {model_type} ({symbol}-{strategy}) acc={acc:.4f}")
    except Exception as e:
        print(f"[ERROR] meta 저장 실패: {e}")

def train_one_model(symbol, strategy, max_epochs=20):
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

        # ✅ 희소 클래스 복제
        X_raw, y_raw = balance_classes(X_raw, y_raw, min_count=20)

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

        # ✅ class_weight 계산 추가
        from collections import Counter
        counts = Counter(y_train)
        total = sum(counts.values())
        class_weight = [total / counts.get(i, 1) for i in range(NUM_CLASSES)]
        class_weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(DEVICE)

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size=input_size, output_size=NUM_CLASSES).to(DEVICE).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # ✅ CrossEntropyLoss with class_weight
            lossfn = nn.CrossEntropyLoss(weight=class_weight_tensor)

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
                del wrong_loader, wrong_ds
                torch.cuda.empty_cache()
                gc.collect()

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

            # ✅ used_feature_columns 저장 추가
            save_model_metadata(
                symbol, strategy, model_type, acc, f1, val_loss,
                input_size=input_size,
                class_counts=Counter(y_train),
                used_feature_columns=list(df_feat.drop(columns=["timestamp"]).columns)
            )

            log_training_result(symbol, strategy, model_type, acc, f1, val_loss)
            torch.save(model.state_dict(), f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.pt")

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

def balance_classes(X, y, min_count=20):
    import numpy as np
    from collections import Counter

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("[❌ balance_classes 실패] X 또는 y 비어있음")
        return X, y

    y = y.astype(np.int64)
    mask = (y != -1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(y) == 0:
        raise Exception("[❌ balance_classes 실패] 라벨 제거 후 샘플 없음")

    class_counts = Counter(y)
    print(f"[🔢 기존 클래스 분포] {dict(class_counts)}")

    try:
        # ✅ SMOTE 적용
        from imblearn.over_sampling import SMOTE
        nsamples, nx, ny = X.shape
        X_reshaped = X.reshape((nsamples, nx * ny))
        smote = SMOTE(random_state=42, sampling_strategy='not majority')
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        X = X_resampled.reshape((-1, nx, ny))
        y = y_resampled
        print(f"[✅ SMOTE 완료] 샘플수: {len(y)}")
    except Exception as e:
        print(f"[⚠️ SMOTE 실패] → fallback 기존 방식 사용: {e}")

        X_balanced, y_balanced = list(X), list(y)
        max_count = max(class_counts.values()) if class_counts else 0
        target_count = max(min_count, int(max_count * 0.8))

        all_classes = range(21)  # NUM_CLASSES = 21
        for cls in all_classes:
            count = class_counts.get(cls, 0)
            needed = max(0, target_count - count)

            if needed > 0:
                indices = [i for i, label in enumerate(y) if label == cls]
                if indices:
                    # ✅ 실제 샘플 기반 + Gaussian noise 추가
                    reps = np.random.choice(indices, needed, replace=True)
                    noisy_samples = X[reps] + np.random.normal(0, 0.01, X[reps].shape).astype(np.float32)
                    X_balanced.extend(noisy_samples)
                    y_balanced.extend(y[reps])
                    print(f"[복제+Noise] 클래스 {cls} → {needed}개 추가")
                else:
                    # ✅ noise sample 제거 (생성하지 않음)
                    print(f"[스킵] 클래스 {cls} → 샘플 없음, noise sample 생성 생략")

        combined = list(zip(X_balanced, y_balanced))
        np.random.shuffle(combined)
        X_shuffled, y_shuffled = zip(*combined)
        return np.array(X_shuffled), np.array(y_shuffled, dtype=np.int64)

    # ✅ SMOTE 성공시 반환
    return np.array(X), np.array(y, dtype=np.int64)


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

