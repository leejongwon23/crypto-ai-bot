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

def train_one_model(symbol, strategy, group_id=None, max_epochs=20):
    import os, gc, traceback, torch, numpy as np, pandas as pd, json
    from datetime import datetime; from collections import Counter
    from ssl_pretrain import masked_reconstruction
    from config import get_FEATURE_INPUT_SIZE, get_class_ranges, get_class_groups, set_NUM_CLASSES  # ✅ set_NUM_CLASSES 추가
    from torch.utils.data import TensorDataset, DataLoader
    from model.base_model import get_model
    from logger import log_training_result, record_model_success
    from data.utils import get_kline_by_strategy, compute_features
    from wrong_data_loader import load_training_prediction_data
    import pytz
    from ranger_adabelief import RangerAdaBelief as Ranger
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import MinMaxScaler

    print("✅ [train_one_model 호출됨]")
    now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = get_FEATURE_INPUT_SIZE()
    group_ids = [group_id] if group_id is not None else list(range(1))

    for gid in group_ids:
        print(f"▶ [학습시작] {symbol}-{strategy}-group{gid}")
        try:
            masked_reconstruction(symbol, strategy, input_size)

            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 100:
                raise Exception("⛔ get_kline 데이터 부족")

            feat = compute_features(symbol, df, strategy)
            if feat is None or len(feat) < 100:
                raise Exception("⛔ feature 데이터 부족")

            features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
            feat_scaled = MinMaxScaler().fit_transform(features_only)

            returns = df["close"].pct_change().fillna(0).values
            class_ranges = get_class_ranges(group_id=gid)
            num_classes = len(class_ranges)  # ✅ 실제 클래스 수

            # ✅ 전역 NUM_CLASSES 업데이트
            set_NUM_CLASSES(num_classes)

            group_classes = get_class_groups(num_classes=num_classes)  # ✅ 동적 클래스 그룹 계산

            labels = []
            for r in returns:
                matched = False
                for i, rng in enumerate(class_ranges):
                    if isinstance(rng, tuple) and len(rng) == 2:
                        low, high = rng
                        if low <= r <= high:
                            labels.append(i)
                            matched = True
                            break
                if not matched:
                    labels.append(0)

            window = 60
            X, y = [], []
            for i in range(len(feat_scaled) - window):
                X.append(feat_scaled[i:i+window])
                y.append(labels[i + window] if i + window < len(labels) else 0)

            X, y = np.array(X), np.array(y)

            fail_X, fail_y = load_training_prediction_data(symbol, strategy, input_size, window, group_id=gid)
            if fail_X is not None and len(fail_X) > 0:
                print(f"📌 실패 샘플 {len(fail_X)}건 추가 병합")
                X = np.concatenate([X, fail_X], axis=0)
                y = np.concatenate([y, fail_y], axis=0)

            if len(X) < 10:
                raise Exception("⛔ 유효한 학습 샘플 부족")

            for model_type in ["lstm", "cnn_lstm", "transformer"]:
                model = get_model(model_type, input_size=input_size, output_size=num_classes)
                if model is None:
                    raise Exception(f"⛔ get_model({model_type}) → None 반환됨")

                model.to(DEVICE)

                model_base = f"{symbol}_{strategy}_{model_type}_group{gid}_cls{num_classes}"
                model_path = os.path.join("/persistent/models", f"{model_base}.pt")

                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                    print(f"🔁 기존 모델 불러와 이어학습 시작: {model_path}")
                else:
                    print(f"🆕 신규 모델 학습 시작: {model_path}")

                optimizer = Ranger(model.parameters(), lr=0.001)
                criterion = torch.nn.CrossEntropyLoss()
                model.train()

                ratio = int(len(X) * 0.8)
                X_train = torch.tensor(X[:ratio], dtype=torch.float32)
                y_train = torch.tensor(y[:ratio], dtype=torch.long)
                X_val = torch.tensor(X[ratio:], dtype=torch.float32)
                y_val = torch.tensor(y[ratio:], dtype=torch.long)

                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

                for epoch in range(max_epochs):
                    total_loss = 0
                    for xb, yb in train_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        optimizer.zero_grad()
                        out = model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    print(f"[{model_type}][Epoch {epoch+1}/{max_epochs}] Loss: {total_loss:.4f}")

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(DEVICE)
                        out = model(xb)
                        preds = torch.argmax(out, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(yb.numpy())

                acc = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='macro')
                print(f"[🎯 {model_type}] acc={acc:.4f}, f1={f1:.4f}")

                if acc == 0.0 and f1 == 0.0:
                    print(f"⛔ {model_type} 평가 실패 → 저장/로깅 생략")
                    continue

                os.makedirs("/persistent/models", exist_ok=True)
                torch.save(model.state_dict(), model_path)

                meta_info = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "model": model_type,
                    "group_id": gid,
                    "num_classes": num_classes,
                    "input_size": input_size,
                    "timestamp": now_kst().isoformat()
                }
                with open(model_path.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta_info, f, ensure_ascii=False, indent=2)

                log_training_result(symbol=symbol, strategy=strategy, model=model_path,
                                    accuracy=acc, f1=f1, loss=total_loss)

                success_flag = acc > 0.6 and f1 > 0.55
                record_model_success(model_base, success_flag)

        except Exception as e:
            print(f"[❌ train_one_model 실패] {symbol}-{strategy}-group{gid} → {e}")
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
    ✅ [YOPO 구조 반영]
    - 각 심볼에 대해 전략별 전체 그룹 학습 실행 (group_id=None)
    - 학습 완료 후 meta 보정, 실패학습, 진화형 메타러너 학습까지 자동 수행
    """
    global training_in_progress
    from telegram_bot import send_message
    import maintenance_fix_meta
    import time

    strategies = ["단기", "중기", "장기"]

    print(f"🚀 [train_models] 심볼 학습 시작: {symbol_list}")

    for symbol in symbol_list:
        print(f"\n🔁 [심볼 시작] {symbol}")

        for strategy in strategies:
            if training_in_progress.get(strategy, False):
                print(f"⚠️ 중복 실행 방지: {strategy}")
                continue

            training_in_progress[strategy] = True
            try:
                # ✅ 핵심: group_id=None → 전체 그룹 자동 반복 학습
                train_one_model(symbol, strategy, group_id=None)
            except Exception as e:
                print(f"[❌ 학습 실패] {symbol}-{strategy} → {e}")
            finally:
                training_in_progress[strategy] = False
                print(f"✅ {symbol}-{strategy} 전체 그룹 학습 완료")
                time.sleep(2)

    # ✅ 메타정보 보정
    try:
        maintenance_fix_meta.fix_all_meta_json()
        print(f"✅ meta 보정 완료: {symbol_list}")
    except Exception as e:
        print(f"[⚠️ meta 보정 실패] {e}")

    # ✅ 실패 학습 루프 실행
    try:
        import failure_trainer
        failure_trainer.run_failure_training()
        print(f"✅ 실패학습 루프 완료")
    except Exception as e:
        print(f"[❌ 실패학습 루프 예외] {e}")

    # ✅ 진화형 메타러너 학습 실행
    try:
        from evo_meta_learner import train_evo_meta
        train_evo_meta()
        print(f"✅ 진화형 메타러너 학습 완료")
    except Exception as e:
        print(f"[❌ 진화형 메타러너 학습 실패] {e}")

    # ✅ 알림 전송
    send_message(f"✅ 전체 심볼 학습 완료: {symbol_list}")


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
    import time, os, json, traceback
    import maintenance_fix_meta
    from data.utils import SYMBOL_GROUPS, _kline_cache, _feature_cache
    from train import train_one_model
    from recommend import main
    import safe_cleanup
    from evo_meta_learner import train_evo_meta_loop, train_evo_meta
    from wrong_data_loader import load_training_prediction_data
    from config import get_FEATURE_INPUT_SIZE, get_class_groups, get_class_ranges
    from failure_db import ensure_failure_db

    ensure_failure_db()
    FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

    FORCE_TRAINING = True  # 필요 시 외부에서 제어
    done_path = "/persistent/train_done.json"
    train_done = {}

    try:
        if os.path.exists(done_path):
            with open(done_path, "r", encoding="utf-8") as f:
                train_done = json.load(f)
            if not isinstance(train_done, dict):
                print("⚠️ train_done 구조 오류 → 초기화")
                train_done = {}
    except Exception as e:
        print(f"⚠️ train_done 파싱 실패 → 초기화: {e}")
        train_done = {}

    loop_count = 0
    group_count = len(SYMBOL_GROUPS)
    print(f"🚀 전체 {group_count}개 그룹 학습 루프 시작")

    while True:
        loop_count += 1
        print(f"\n🔄 전체 그룹 순회 루프 #{loop_count} 시작")

        for idx, group in enumerate(SYMBOL_GROUPS):
            print(f"\n🚀 [그룹 {idx}/{group_count}] 학습 시작 | 심볼 수: {len(group)}")
            _kline_cache.clear()
            _feature_cache.clear()

            for symbol in group:
                for strategy in ["단기", "중기", "장기"]:
                    train_done.setdefault(symbol, {}).setdefault(strategy, {})

                    try:
                        num_classes = len(get_class_ranges(symbol=symbol, strategy=strategy))
                        class_groups = get_class_groups(num_classes=num_classes)
                        MAX_GROUP_ID = len(class_groups) - 1
                    except Exception as e:
                        print(f"[⚠️ 동적 클래스 계산 실패] {symbol}-{strategy} → {e}")
                        continue

                    all_success = True

                    for gid in range(MAX_GROUP_ID + 1):
                        if not FORCE_TRAINING and train_done[symbol][strategy].get(str(gid), False):
                            print(f"[⏭️ 스킵] {symbol}-{strategy}-group{gid} (이미 학습됨)")
                            continue

                        try:
                            train_one_model(symbol, strategy, group_id=gid)
                            train_done[symbol][strategy][str(gid)] = True
                            with open(done_path, "w", encoding="utf-8") as f:
                                json.dump(train_done, f, ensure_ascii=False, indent=2)
                            print(f"[✅ 학습 완료] {symbol}-{strategy}-group{gid}")
                        except Exception as e:
                            print(f"[❌ 학습 실패] {symbol}-{strategy}-group{gid} → {e}")
                            traceback.print_exc()
                            all_success = False

                    # 그룹 전체 학습 완료 여부 엄격 검증
                    group_all_trained = all(
                        train_done[symbol][strategy].get(str(gid), False)
                        for gid in range(MAX_GROUP_ID + 1)
                    )

                    if all_success and group_all_trained:
                        try:
                            print(f"[▶ 예측 시도] {symbol}-{strategy} (모든 그룹 학습 완료)")
                            main(symbol=symbol, strategy=strategy, force=True, allow_prediction=True)
                            print(f"[✅ 예측 완료] {symbol}-{strategy}")

                            # 실패 데이터 로딩 후 이어학습/메타러너 학습
                            try:
                                X, y = load_training_prediction_data(
                                    symbol, strategy,
                                    input_size=FEATURE_INPUT_SIZE,
                                    window=20,
                                    group_id=None
                                )
                                if X is not None and y is not None and len(X) > 0:
                                    print(f"[▶ 이어학습 시작] {symbol}-{strategy}")
                                    train_one_model(symbol, strategy, group_id=None)
                                    print(f"[✅ 이어학습 완료] {symbol}-{strategy}")

                                    train_evo_meta(X, y, FEATURE_INPUT_SIZE)
                                    print(f"[✅ 진화형 메타러너 학습 완료] {symbol}-{strategy}")
                                else:
                                    print(f"[⚠️ 이어학습/메타러너 스킵] 데이터 부족: {symbol}-{strategy}")
                            except Exception as e:
                                print(f"[⚠️ 이어학습/메타러너 실패] {symbol}-{strategy} → {e}")

                        except Exception as e:
                            print(f"[❌ 예측 실패] {symbol}-{strategy} → {e}")
                            traceback.print_exc()

            try:
                maintenance_fix_meta.fix_all_meta_json()
                safe_cleanup.auto_delete_old_logs()
            except Exception as e:
                print(f"[⚠️ 그룹 {idx} 후처리 실패] → {e}")

            print(f"🕒 그룹 {idx} 완료 → {delay_minutes}분 대기")
            time.sleep(delay_minutes * 60)

        try:
            train_evo_meta_loop()
        except Exception as e:
            print(f"[⚠️ 진화형 메타러너 루프 학습 실패] → {e}")


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
