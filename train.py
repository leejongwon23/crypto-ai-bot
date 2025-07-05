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
    from ssl_pretrain import masked_reconstruction
    print(f"▶ 학습 시작: {symbol}-{strategy}")

    try:
        # ✅ SSL pretraining 실행 (input_size=14 고정)
        masked_reconstruction(symbol, strategy, input_size=14, mask_ratio=0.2, epochs=5)

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

        # ✅ input_size=14 강제 적용
        X_raw, y_raw = create_dataset(df_feat.to_dict(orient="records"), window=window, strategy=strategy, input_size=14)
        if X_raw is None or y_raw is None or len(X_raw) < 5:
            print("⛔ 중단: 학습 데이터 부족")
            return

        print("[INFO] balance_classes(min_count=30) 호출")
        X_raw, y_raw = balance_classes(X_raw, y_raw, min_count=30)

        y_raw, X_raw = np.array(y_raw), np.array(X_raw, dtype=np.float32)
        mask = (y_raw >= 0) & (y_raw < NUM_CLASSES)
        X_raw, y_raw = X_raw[mask], y_raw[mask]
        if len(X_raw) < 5:
            print("⛔ 중단: 유효 샘플 부족")
            return

        input_size = 14  # ✅ input_size 고정
        val_len = max(5, int(len(X_raw) * 0.2))

        # ✅ Curriculum Learning
        sorted_idx = np.argsort(y_raw)
        X_raw, y_raw = X_raw[sorted_idx], y_raw[sorted_idx]

        X_train, y_train, X_val, y_val = X_raw[:-val_len], y_raw[:-val_len], X_raw[-val_len:], y_raw[-val_len:]

        wrong_data = load_training_prediction_data(symbol, strategy, input_size, window)
        wrong_ds = TensorDataset(torch.tensor([x for x, _ in wrong_data], dtype=torch.float32),
                                 torch.tensor([y for _, y in wrong_data], dtype=torch.long)) if wrong_data else None

        from collections import Counter
        counts = Counter(y_train)
        total = sum(counts.values())
        class_weight = [total / counts.get(i, 1) for i in range(NUM_CLASSES)]
        class_weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(DEVICE)

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size=input_size, output_size=NUM_CLASSES).to(DEVICE).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossfn = nn.CrossEntropyLoss(weight=class_weight_tensor)

            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.long))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

            # ✅ Active Sampling
            for epoch in range(max_epochs):
                indices = np.random.choice(len(X_train), int(len(X_train)*0.8), replace=False)
                sampled_X = X_train[indices]
                sampled_y = y_train[indices]

                sampled_ds = TensorDataset(torch.tensor(sampled_X, dtype=torch.float32),
                                           torch.tensor(sampled_y, dtype=torch.long))
                sampled_loader = DataLoader(sampled_ds, batch_size=32, shuffle=True, num_workers=2)

                for xb, yb in sampled_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = lossfn(logits, yb)
                    if torch.isfinite(loss):
                        optimizer.zero_grad(); loss.backward(); optimizer.step()

            # ✅ 실패 집중 학습
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
    from imblearn.over_sampling import SMOTE
    from logger import log_prediction  # ✅ logger import 추가

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

    nsamples, nx, ny = X.shape
    X_balanced, y_balanced = list(X), list(y)

    max_count = max(class_counts.values()) if class_counts else min_count
    target_count = max(min_count, int(max_count * 0.8))

    for cls in range(21):  # NUM_CLASSES = 21
        indices = [i for i, label in enumerate(y) if label == cls]
        count = len(indices)
        needed = max(0, target_count - count)

        if needed > 0:
            if count >= 2:
                try:
                    X_cls = X[indices].reshape((count, nx * ny))
                    k_neighbors = min(count - 1, 5)
                    smote = SMOTE(random_state=42, sampling_strategy={cls: count + needed}, k_neighbors=k_neighbors)
                    X_res, y_res = smote.fit_resample(X_cls, np.array([cls]*count))
                    X_new = X_res[count:].reshape((-1, nx, ny))
                    if len(X_new) > needed:
                        X_new = X_new[:needed]
                    X_balanced.extend(X_new)
                    y_balanced.extend([cls]*len(X_new))
                    print(f"[✅ SMOTE 성공] 클래스 {cls} → {len(X_new)}개 추가")

                    log_prediction(
                        symbol="augmentation", strategy="augmentation",
                        direction=f"SMOTE-{cls}", entry_price=0, target_price=0,
                        model="augmentation", success=True,
                        reason=f"SMOTE {cls} {len(X_new)}개 추가",
                        rate=0.0, timestamp=None, return_value=0.0,
                        volatility=False, source="augmentation",
                        predicted_class=cls, label=cls, augmentation="smote"
                    )

                except Exception as e:
                    print(f"[⚠️ SMOTE 실패] 클래스 {cls} → fallback: {e}")
                    reps = np.random.choice(indices, needed, replace=True)
                    noisy_samples = X[reps] + np.random.normal(0, 0.05, X[reps].shape).astype(np.float32)
                    
                    # ✅ 추가: Noise + Mixup + Time Masking
                    mixup_samples = noisy_samples.copy()
                    for i in range(len(mixup_samples)):
                        j = np.random.randint(len(X))
                        lam = np.random.beta(0.2, 0.2)
                        mixup_samples[i] = lam * mixup_samples[i] + (1 - lam) * X[j]

                        # Time Masking
                        t = np.random.randint(0, nx)
                        mixup_samples[i][t] = 0.0

                    X_balanced.extend(mixup_samples)
                    y_balanced.extend([cls]*needed)
                    print(f"[복제+Noise+Mixup+Masking] 클래스 {cls} → {needed}개 추가")
            elif count == 1:
                reps = np.repeat(indices[0], needed)
                noisy_samples = X[reps] + np.random.normal(0, 0.05, X[reps].shape).astype(np.float32)
                X_balanced.extend(noisy_samples)
                y_balanced.extend([cls]*needed)
                print(f"[복제+Noise] 클래스 {cls} → {needed}개 추가 (1개 복제)")
            else:
                print(f"[스킵] 클래스 {cls} → 샘플 없음, noise sample 생성 생략")

    combined = list(zip(X_balanced, y_balanced))
    np.random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)

    final_counts = Counter(y_shuffled)
    print(f"[📊 최종 클래스 분포] {dict(final_counts)}")
    print(f"[✅ balance_classes 완료] 최종 샘플수: {len(y_shuffled)}")

    return np.array(X_shuffled), np.array(y_shuffled, dtype=np.int64)

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
def pretrain_ssl_features(symbol, strategy, pretrain_epochs=5):
    """
    ✅ [설명] Self-Supervised Learning pretraining
    - feature reconstruction 기반 사전학습
    """
    from model.base_model import get_model

    print(f"▶ SSL Pretraining 시작: {symbol}-{strategy}")

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print("⛔ 중단: 시세 데이터 없음")
        return

    df_feat = compute_features(symbol, df, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        print("⛔ 중단: 피처 생성 실패 또는 NaN")
        return

    features_only = df_feat.drop(columns=["timestamp"], errors="ignore")
    feat_scaled = MinMaxScaler().fit_transform(features_only)
    X = np.expand_dims(feat_scaled, axis=1)  # (samples, 1, features)

    input_size = X.shape[2]
    model = get_model("autoencoder", input_size=input_size, output_size=input_size).to(DEVICE).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    for epoch in range(pretrain_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = lossfn(out, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[SSL Pretrain {epoch+1}/{pretrain_epochs}] loss={avg_loss:.6f}")

    torch.save(model.state_dict(), f"{MODEL_DIR}/{symbol}_{strategy}_ssl_pretrain.pt")
    print(f"✅ SSL Pretraining 완료: {symbol}-{strategy}")
