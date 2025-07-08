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
from config import FEATURE_INPUT_SIZE


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

    if used_feature_columns:
        meta["used_feature_columns"] = used_feature_columns

    path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}.meta.json"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # ✅ 경로 없을 시 자동생성
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[메타저장] {model_type} ({symbol}-{strategy}) acc={acc:.4f}")
    except Exception as e:
        print(f"[ERROR] meta 저장 실패: {e}")

def get_class_groups(num_classes=21, group_size=7):
    """
    ✅ 클래스 그룹화 함수 (YOPO v4.1)
    - num_classes를 group_size 크기로 나누어 그룹화
    - num_classes ≤ group_size 시 단일 그룹 반환
    - ex) num_classes=21, group_size=7 → [[0-6], [7-13], [14-20]]
    """
    if num_classes <= group_size:
        return [list(range(num_classes))]
    return [list(range(i, min(i+group_size, num_classes))) for i in range(0, num_classes, group_size)]

def train_one_model(symbol, strategy, max_epochs=20):
    import os, gc
    from focal_loss import FocalLoss
    from ssl_pretrain import masked_reconstruction
    from window_optimizer import find_best_windows
    from config import FEATURE_INPUT_SIZE
    from collections import Counter
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import random
    from regime_change_detection import detect_regime_change
    from meta_learning import maml_train_entry
    from model.base_model import get_model
    from logger import log_training_result  # ✅ 추가

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ 학습 시작: {symbol}-{strategy}")

    try:
        df_regime = get_kline_by_strategy(symbol, strategy)
        if df_regime is not None and not df_regime.empty:
            if detect_regime_change(df_regime):
                print(f"[info] {symbol}-{strategy} regime change detected → meta-learning 권장")

        masked_reconstruction(symbol, strategy, input_size=FEATURE_INPUT_SIZE, mask_ratio=0.2, epochs=5)

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print("⛔ 중단: 시세 데이터 없음")
            return

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
            print("⛔ 중단: 피처 생성 실패 또는 NaN")
            return

        window_list = find_best_windows(symbol, strategy)
        features_only = df_feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        input_size = features_only.shape[1]

        if input_size < FEATURE_INPUT_SIZE:
            for pad_col in range(input_size, FEATURE_INPUT_SIZE):
                df_feat[f"pad_{pad_col}"] = 0.0
            input_size = FEATURE_INPUT_SIZE
            print(f"[info] input_size padded to FEATURE_INPUT_SIZE={FEATURE_INPUT_SIZE}")

        print(f"[info] train_one_model input_size: {input_size}")

        class_groups = get_class_groups()

        for window in window_list:
            try:
                X_raw, y_raw = create_dataset(df_feat.to_dict(orient="records"), window=window, strategy=strategy, input_size=input_size)
                if X_raw is None or y_raw is None or len(X_raw) < 5:
                    print(f"⛔ 중단: window={window} 학습 데이터 부족")
                    continue

                if X_raw.shape[2] != input_size:
                    print(f"[❌ 오류] feature input_size 불일치: X_raw.shape[2]={X_raw.shape[2]} vs input_size={input_size}")
                    continue

                val_len = max(5, int(len(X_raw) * 0.2))
                sorted_idx = np.argsort(y_raw)
                X_raw, y_raw = X_raw[sorted_idx], y_raw[sorted_idx]
                X_train, y_train, X_val, y_val = X_raw[:-val_len], y_raw[:-val_len], X_raw[-val_len:], y_raw[-val_len:]

                for group_id, group_classes in enumerate(class_groups):
                    group_mask = np.isin(y_train, group_classes)
                    X_train_group = X_train[group_mask]
                    y_train_group = y_train[group_mask]

                    if len(y_train_group) < 2:
                        print(f"[⚠️ 스킵] window={window} group-{group_id}: 학습 데이터 부족 ({len(y_train_group)}) → 전체 모델 스킵")
                        continue

                    output_size = len(group_classes)
                    if output_size == 0:
                        print(f"[⚠️ 스킵] group-{group_id} output_size=0 → 모델 학습 스킵")
                        continue

                    # ✅ [Step 4-1] validation 라벨 재인코딩 추가
                    val_mask = np.isin(y_val, group_classes)
                    X_val_group = X_val[val_mask]
                    y_val_group = np.array([group_classes.index(y) for y in y_val[val_mask]])

                    for model_type in ["lstm", "cnn_lstm", "transformer"]:
                        target_count = 50
                        repeat_factor = max(1, int(np.ceil(target_count / len(y_train_group))))

                        X_train_group, y_train_group = augment_and_expand(X_train_group, y_train_group, repeat_factor, group_classes, target_count)

                        counts_group = Counter(y_train_group)
                        total_group = sum(counts_group.values())
                        class_weight_group = [total_group / counts_group.get(i, 1) for i in range(output_size)]

                        if len(class_weight_group) != output_size:
                            print(f"[❌ 오류] class_weight_group 길이 불일치: {len(class_weight_group)} != output_size {output_size}")
                            class_weight_group = [1.0 for _ in range(output_size)]

                        class_weight_tensor = torch.tensor(class_weight_group, dtype=torch.float32).to(DEVICE)

                        model = get_model(model_type, input_size=input_size, output_size=output_size).to(DEVICE).train()
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                        lossfn_ce = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

                        train_ds = TensorDataset(
                            torch.tensor(X_train_group, dtype=torch.float32),
                            torch.tensor(y_train_group, dtype=torch.long)
                        )
                        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

                        for epoch in range(max_epochs):
                            for xb, yb in train_loader:
                                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                                logits = model(xb)
                                loss = lossfn_ce(logits, yb)

                                if torch.isfinite(loss):
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            val_inputs = torch.tensor(X_val_group, dtype=torch.float32).to(DEVICE)
                            val_labels = torch.tensor(y_val_group, dtype=torch.long).to(DEVICE)
                            val_logits = model(val_inputs)
                            val_preds = torch.argmax(val_logits, dim=1)
                            val_acc = (val_preds == val_labels).float().mean().item()
                            print(f"[📈 validation accuracy] {symbol}-{strategy}-{model_type} acc={val_acc:.4f}")

                        # ✅ 학습 로그 기록 추가
                        log_training_result(symbol, strategy, model_type, acc=val_acc, f1=0.0, loss=float(loss.item()))

                        meta = {
                            "symbol": symbol, "strategy": strategy, "model": model_type,
                            "group_id": group_id, "window": window,
                            "input_size": input_size,
                            "val_accuracy": float(round(val_acc, 4)),
                            "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        meta_path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}_group{group_id}_window{window}.meta.json"
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2, ensure_ascii=False)

                        model_path = f"{MODEL_DIR}/{symbol}_{strategy}_{model_type}_group{group_id}_window{window}.pt"
                        torch.save(model.state_dict(), model_path)

                        print(f"[✅ 저장 완료] {model_type} group-{group_id} window-{window}")

                        del model, xb, yb, logits
                        torch.cuda.empty_cache()
                        gc.collect()

                val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=32)
                train_loader = DataLoader(train_ds, batch_size=32)

                if 'model' in locals():
                    maml_train_entry(model, train_loader, val_loader)
                else:
                    print("[⚠️ model 없음 → maml_train_entry 스킵]")

            except Exception as e:
                print(f"[ERROR] window={window}: {e}")

    except Exception as e:
        print(f"[ERROR] {symbol}-{strategy}: {e}")



# ✅ augmentation 함수 추가
def augment_and_expand(X_train_group, y_train_group, repeat_factor, group_classes, target_count):
    import numpy as np
    import random
    from data_augmentation import add_gaussian_noise, apply_scaling, apply_shift, apply_dropout_mask

    X_aug, y_aug = [], []

    per_class_target = max(1, target_count // len(group_classes))

    for cls in group_classes:
        cls_indices = np.where(y_train_group == cls)[0]

        if len(cls_indices) == 0:
            dummy = np.random.normal(0, 1, (per_class_target, X_train_group.shape[1], X_train_group.shape[2])).astype(np.float32)
            X_cls_aug = dummy
            y_cls_aug = np.array([cls] * per_class_target, dtype=np.int64)
        else:
            X_cls = X_train_group[cls_indices]
            y_cls = y_train_group[cls_indices]

            n_repeat = int(np.ceil(per_class_target / len(cls_indices)))
            X_cls_oversampled = np.tile(X_cls, (n_repeat, 1, 1))[:per_class_target]
            y_cls_oversampled = np.tile(y_cls, n_repeat)[:per_class_target]

            X_cls_aug = []
            for x in X_cls_oversampled:
                x1 = add_gaussian_noise(x)
                x2 = apply_scaling(x1)
                x3 = apply_shift(x2)
                x4 = apply_dropout_mask(x3)
                X_cls_aug.append(x4)
            X_cls_aug = np.array(X_cls_aug, dtype=np.float32)
            y_cls_aug = y_cls_oversampled

        X_aug.append(X_cls_aug)
        y_aug.append(y_cls_aug)

    X_aug = np.concatenate(X_aug, axis=0)
    y_aug = np.concatenate(y_aug, axis=0)

    if len(X_aug) < target_count:
        idx = np.random.choice(len(X_aug), target_count - len(X_aug))
        X_aug = np.concatenate([X_aug, X_aug[idx]], axis=0)
        y_aug = np.concatenate([y_aug, y_aug[idx]], axis=0)
    else:
        X_aug = X_aug[:target_count]
        y_aug = y_aug[:target_count]

    # ✅ 라벨 재인코딩 with debug
    y_encoded = []
    for y in y_aug:
        try:
            encoded = group_classes.index(y)
            y_encoded.append(encoded)
        except ValueError:
            print(f"[❌ 라벨 재인코딩 오류] {y} not in group_classes → 제거")
            y_encoded.append(-1)

    # -1 라벨 제거
    X_encoded = X_aug[np.array(y_encoded) != -1]
    y_encoded = np.array([y for y in y_encoded if y != -1])

    # ✅ 디버그 출력
    from collections import Counter
    print(f"[✅ augment_and_expand] 최종 샘플 수: {len(y_encoded)}, 라벨 분포: {Counter(y_encoded)}")

    return X_encoded, y_encoded




def balance_classes(X, y, min_count=20, num_classes=21):
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

    nsamples, nx, ny = X.shape
    X_balanced, y_balanced = list(X), list(y)

    max_count = max(class_counts.values()) if class_counts else min_count
    target_count = max(min_count, int(max_count * 0.8))

    for cls in range(num_classes):
        indices = [i for i, label in enumerate(y) if label == cls]
        count = len(indices)
        needed = max(0, target_count - count)

        if needed > 0:
            if count >= 1:
                reps = np.random.choice(indices, needed, replace=True)
                noisy_samples = X[reps] + np.random.normal(0, 0.05, X[reps].shape).astype(np.float32)

                # ✅ Noise + Mixup + Time Masking
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

                # ✅ 값 범위 이상치 검증 로그 추가
                if np.any(np.isnan(mixup_samples)) or np.any(np.isinf(mixup_samples)):
                    print(f"[⚠️ 경고] 클래스 {cls} 복제 중 NaN 또는 Inf 발생")
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
    - 각 그룹 학습 후 meta 보정, 해당 그룹 심볼만 예측 실행
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

                # ✅ 예측 실행: 해당 그룹 심볼만 예측하도록 수정
                for symbol in group:
                    for strategy in ["단기", "중기", "장기"]:
                        try:
                            from recommend import main
                            main(symbol=symbol, strategy=strategy, force=True, allow_prediction=True)
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
