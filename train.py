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
import time
from data_augmentation import balance_classes
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE
NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
from config import get_class_groups


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
    import os, gc, traceback, torch, json
    from focal_loss import FocalLoss
    from ssl_pretrain import masked_reconstruction
    from window_optimizer import find_best_windows
    from config import get_FEATURE_INPUT_SIZE, get_class_groups
    from collections import Counter
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    from model.base_model import get_model
    from logger import log_training_result
    from data_augmentation import balance_classes
    from wrong_data_loader import load_training_prediction_data
    from datetime import datetime
    import pytz
    from meta_learning import maml_train_entry
    from ranger_adabelief import RangerAdaBelief as Ranger
    from feature_importance import compute_feature_importance, drop_low_importance_features
    from data.utils import get_kline_by_strategy, compute_features, create_dataset

    print("✅ [train_one_model 호출됨]")
    now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = get_FEATURE_INPUT_SIZE()
    print(f"▶ [학습시작] {symbol}-{strategy}")

    try:
        masked_reconstruction(symbol, strategy, input_size=input_size, mask_ratio=0.2, epochs=5)
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print(f"⛔ [중단] {symbol}-{strategy}: 시세 데이터 없음")
            return

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
            print(f"⛔ [중단] {symbol}-{strategy}: 피처 생성 실패 또는 NaN")
            return

        try:
            feature_names = [col for col in df_feat.columns if col not in ["timestamp", "strategy"]]
            dummy_X = torch.tensor(np.random.rand(10, 20, input_size), dtype=torch.float32).to(DEVICE)
            dummy_y = torch.randint(0, 2, (10,), dtype=torch.long).to(DEVICE)
            dummy_model = get_model("lstm", input_size=input_size, output_size=2).to(DEVICE)
            importances = compute_feature_importance(dummy_model, dummy_X, dummy_y, feature_names, method="baseline")
            df_feat = drop_low_importance_features(df_feat, importances, threshold=0.01, min_features=5)
            print("[✅ feature drop 완료] 중요도가 낮은 feature 제거됨")
        except Exception as e:
            print(f"[⚠️ feature drop 실패] {e}")

        window_list = find_best_windows(symbol, strategy)
        if not window_list:
            print(f"⛔ [중단] {symbol}-{strategy}: window_list 없음")
            return

        print(f"✅ [진행] {symbol}-{strategy}: window_list={window_list}")
        class_groups = get_class_groups()
        trained_any = False

        for window in window_list:
            X_raw, y_raw = create_dataset(df_feat.to_dict(orient="records"), window=window, strategy=strategy, input_size=input_size)

            if X_raw is None or y_raw is None or len(X_raw) == 0:
                print(f"[⚠️ create_dataset 실패 또는 없음 → dummy 생성] {symbol}-{strategy}, window={window}")
                dummy_shape = (10, window, input_size)
                X_raw = np.random.normal(0, 1, size=dummy_shape).astype(np.float32)
                y_raw = np.random.randint(0, len(class_groups[0]), size=(10,))

            fail_X, fail_y = load_training_prediction_data(symbol, strategy, input_size=input_size, window=window)
            if fail_X is not None and fail_y is not None and len(fail_X) > 0:
                X_raw = np.concatenate([X_raw, fail_X], axis=0)
                y_raw = np.concatenate([y_raw, fail_y], axis=0)
                print(f"[🔁 실패재학습 추가] {len(fail_X)}개 실패샘플 합산 완료")

            val_len = max(5, int(len(X_raw) * 0.2))
            if len(X_raw) <= val_len:
                val_indices = np.random.choice(len(X_raw), val_len, replace=True)
                X_val, y_val = X_raw[val_indices], y_raw[val_indices]
                X_train, y_train = X_raw, y_raw
            else:
                X_train, y_train, X_val, y_val = X_raw[:-val_len], y_raw[:-val_len], X_raw[-val_len:], y_raw[-val_len:]

            for group_id, group_classes in enumerate(class_groups):
                train_mask = np.isin(y_train, group_classes)
                X_train_group = X_train[train_mask]
                y_train_group = y_train[train_mask]

                if len(y_train_group) < 2:
                    print(f"⚠️ [데이터 부족 → dummy 보강] group_id={group_id}")
                    dummy_count = 10
                    dummy_x = np.random.normal(0, 1, size=(dummy_count, window, input_size)).astype(np.float32)
                    dummy_y = np.random.randint(0, len(group_classes), size=(dummy_count,))
                    X_train_group = np.concatenate([X_train_group, dummy_x], axis=0)
                    y_train_group = np.concatenate([y_train_group, dummy_y], axis=0)

                X_train_group, y_train_group = balance_classes(X_train_group, y_train_group, min_count=20, num_classes=len(group_classes))
                print(f"[📊 학습 데이터 클래스 분포] {Counter(y_train_group)}")

                val_mask = np.isin(y_val, group_classes)
                X_val_group = X_val[val_mask]
                y_val_group = y_val[val_mask]
                if len(y_val_group) == 0:
                    X_val_group = np.random.normal(0, 1, size=(10, window, input_size)).astype(np.float32)
                    y_val_group = np.random.randint(0, len(group_classes), size=(10,))

                y_train_group = np.array([group_classes.index(y) for y in y_train_group])
                y_val_group = np.array([group_classes.index(y) for y in y_val_group])

                model_type = "cnn_lstm"
                lr = 1e-4
                batch_size = 32
                hidden_size = 64
                dropout = 0.3
                opt_type = "AdamW"
                loss_type = "FocalLoss"

                model = get_model(model_type, input_size=input_size, output_size=len(group_classes)).to(DEVICE).train()
                if hasattr(model, "set_hyperparams"):
                    model.set_hyperparams(hidden_size=hidden_size, dropout=dropout)

                optimizer = {
                    "Adam": torch.optim.Adam(model.parameters(), lr=lr),
                    "AdamW": torch.optim.AdamW(model.parameters(), lr=lr),
                    "Ranger": Ranger(model.parameters(), lr=lr)
                }.get(opt_type, torch.optim.Adam(model.parameters(), lr=lr))

                lossfn = FocalLoss() if loss_type == "FocalLoss" else torch.nn.CrossEntropyLoss()

                X_train_tensor = torch.tensor(X_train_group[:, -1, :], dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_group, dtype=torch.long)
                X_val_tensor = torch.tensor(X_val_group[:, -1, :], dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val_group, dtype=torch.long)

                train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

                for epoch in range(max_epochs):
                    for xb, yb in train_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        logits = model(xb)
                        loss = lossfn(logits, yb)
                        if torch.isfinite(loss):
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                maml_loss = maml_train_entry(model, train_loader, val_loader, inner_lr=0.01, outer_lr=0.001, inner_steps=1)
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_tensor.to(DEVICE))
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = (val_preds == y_val_tensor.to(DEVICE)).float().mean().item()

                model_path = f"/persistent/models/{symbol}_{strategy}_{model_type}_{opt_type}_{loss_type}_lr{lr}_bs={batch_size}_hs={hidden_size}_dr={dropout}_group{group_id}_window{window}.pt"
                torch.save(model.state_dict(), model_path)

                log_training_result(symbol, strategy, f"{model_type}_{opt_type}_{loss_type}_lr{lr}_bs={batch_size}_hs={hidden_size}_dr={dropout}", acc=val_acc, f1=0.0, loss=loss.item())
                print(f"[✅ 학습완료] {symbol}-{strategy} group:{group_id} acc:{val_acc:.4f}")
                trained_any = True

                del model, optimizer, lossfn, train_loader, val_loader
                torch.cuda.empty_cache()
                gc.collect()

        if not trained_any:
            log_training_result(symbol, strategy, "학습실패", acc=0.0, f1=0.0, loss=0.0)
            print(f"[❌ 학습실패 기록] {symbol}-{strategy}")

    except Exception as e:
        print(f"[ERROR] {symbol}-{strategy}: {e}")
        traceback.print_exc()



# ✅ augmentation 함수 추가
def augment_and_expand(X_train_group, y_train_group, repeat_factor, group_classes, target_count):
    import numpy as np
    import random
    from data_augmentation import add_gaussian_noise, apply_scaling, apply_shift, apply_dropout_mask

    X_aug, y_aug = [], []

    # ✅ 클래스별 개수 계산
    class_counts = {cls: np.sum(y_train_group == cls) for cls in group_classes}
    max_count = max(class_counts.values()) if class_counts else 1
    per_class_target = int(max_count * 0.8)  # 🔥 최대 클래스의 80%로 통일

    for cls in group_classes:
        cls_indices = np.where(y_train_group == cls)[0]

        if len(cls_indices) == 0:
            # ✅ 해당 클래스 샘플 없으면 random noise + 안전 라벨 부여
            dummy = np.random.normal(0, 1, (per_class_target, X_train_group.shape[1], X_train_group.shape[2])).astype(np.float32)
            X_cls_aug = dummy
            y_cls_aug = np.array([cls] * per_class_target, dtype=np.int64)
        else:
            X_cls = X_train_group[cls_indices]
            y_cls = y_train_group[cls_indices]

            # 🔁 부족분 복제 + augmentation (noise + scaling + shift + dropout)
            n_repeat = int(np.ceil(per_class_target / len(cls_indices)))
            X_cls_oversampled = np.tile(X_cls, (n_repeat, 1, 1))[:per_class_target]
            y_cls_oversampled = np.tile(y_cls, n_repeat)[:per_class_target]

            X_cls_aug = []
            for x in X_cls_oversampled:
                x1 = add_gaussian_noise(x)
                x2 = apply_scaling(x1)
                x3 = apply_shift(x2)
                x4 = apply_dropout_mask(x3)
                # ✅ mixup 추가 (간단 믹스업 – 자기 자신 + noise)
                mixup_factor = np.random.uniform(0.7, 1.0)
                x4 = x4 * mixup_factor + np.random.normal(0, 0.05, x4.shape).astype(np.float32) * (1 - mixup_factor)
                X_cls_aug.append(x4)

            X_cls_aug = np.array(X_cls_aug, dtype=np.float32)
            y_cls_aug = y_cls_oversampled

        X_aug.append(X_cls_aug)
        y_aug.append(y_cls_aug)

    X_aug = np.concatenate(X_aug, axis=0)
    y_aug = np.concatenate(y_aug, axis=0)

    # ✅ 최종 target_count 도달 보장
    if len(X_aug) < target_count:
        idx = np.random.choice(len(X_aug), target_count - len(X_aug))
        X_aug = np.concatenate([X_aug, X_aug[idx]], axis=0)
        y_aug = np.concatenate([y_aug, y_aug[idx]], axis=0)
    else:
        X_aug = X_aug[:target_count]
        y_aug = y_aug[:target_count]

    # ✅ 라벨 재인코딩
    y_encoded = []
    X_encoded = []
    for i, y in enumerate(y_aug):
        try:
            encoded = group_classes.index(y)
            y_encoded.append(encoded)
            X_encoded.append(X_aug[i])
        except ValueError:
            print(f"[❌ 라벨 재인코딩 오류] {y} not in group_classes → 제거")
            continue

    X_encoded = np.array(X_encoded, dtype=np.float32)
    y_encoded = np.array(y_encoded, dtype=np.int64)

    # ✅ 디버그 출력
    from collections import Counter
    print(f"[✅ augment_and_expand] 최종 샘플 수: {len(y_encoded)}, 라벨 분포: {Counter(y_encoded)}")

    return X_encoded, y_encoded


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
    ✅ [개선 설명]
    - train_models: 학습만 수행 (예측 실행 제거)
    - 예측 실행은 train_symbol_group_loop에서 그룹별로 호출
    """
    global training_in_progress
    from telegram_bot import send_message
    import maintenance_fix_meta

    strategies = ["단기", "중기", "장기"]

    print(f"🚀 [train_models] 심볼 그룹 학습 시작: {symbol_list}")

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

        time.sleep(2)

    # ✅ meta 보정만 실행
    try:
        maintenance_fix_meta.fix_all_meta_json()
        print(f"✅ meta 보정 완료: 그룹 {symbol_list}")
    except Exception as e:
        print(f"[⚠️ meta 보정 실패] {e}")

    send_message(f"✅ 그룹 학습 완료: {symbol_list}")

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
    ✅ [개선] 루프 순서 강제 초기화 + 로그 개선
    """
    import time
    import maintenance_fix_meta
    from data.utils import SYMBOL_GROUPS, _kline_cache, _feature_cache

    group_count = len(SYMBOL_GROUPS)
    print(f"🚀 전체 {group_count}개 그룹 학습 루프 시작")

    loop_count = 0
    while True:
        loop_count += 1
        print(f"\n🔄 그룹 학습 루프 #{loop_count} 시작")

        # ✅ 순서 강제 초기화
        for idx, group in enumerate(SYMBOL_GROUPS):
            print(f"\n🚀 [그룹 {idx}/{group_count}] 학습 시작 | 심볼: {group}")

            _kline_cache.clear()
            _feature_cache.clear()
            print("[✅ cache cleared] _kline_cache, _feature_cache")

            try:
                train_models(group)
                print(f"[✅ 그룹 {idx}] 학습 완료")

                maintenance_fix_meta.fix_all_meta_json()
                print(f"[✅ meta 보정 완료] 그룹 {idx}")

                # ✅ 예측 실행: 각 그룹 학습 후 예측만 실행
                for symbol in group:
                    for strategy in ["단기", "중기", "장기"]:
                        try:
                            from recommend import main
                            main(symbol=symbol, strategy=strategy, force=True, allow_prediction=True)
                            print(f"[✅ 예측 완료] {symbol}-{strategy}")
                        except Exception as e:
                            print(f"[❌ 예측 실패] {symbol}-{strategy} → {e}")

                print(f"🕒 그룹 {idx} 루프 완료 → {delay_minutes}분 대기")
                time.sleep(delay_minutes * 60)

            except Exception as e:
                print(f"[❌ 그룹 {idx} 루프 중 오류] {e}")
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
