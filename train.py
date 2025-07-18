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
    import os, gc, traceback, torch, json, numpy as np, pandas as pd
    from datetime import datetime; from collections import Counter
    from ssl_pretrain import masked_reconstruction
    from config import get_FEATURE_INPUT_SIZE, get_class_groups
    from torch.utils.data import TensorDataset, DataLoader
    from model.base_model import get_model
    from logger import log_training_result
    from data_augmentation import balance_classes
    from wrong_data_loader import load_training_prediction_data
    from feature_importance import compute_feature_importance, drop_low_importance_features
    from ranger_adabelief import RangerAdaBelief as Ranger
    from window_optimizer import find_best_windows
    from data.utils import get_kline_by_strategy, compute_features, create_dataset
    from focal_loss import FocalLoss
    from meta_learning import maml_train_entry
    import pytz

    now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = get_FEATURE_INPUT_SIZE()
    class_groups_list = get_class_groups()

    print(f"✅ [train_one_model 호출됨] ▶ [학습시작] {symbol}-{strategy}-group{group_id}")
    trained_any = False

    try:
        masked_reconstruction(symbol, strategy, input_size=input_size, mask_ratio=0.2, epochs=5)

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print("[⚠️ get_kline_by_strategy 결과 없음 → dummy로 대체]")
            df = pd.DataFrame([{"timestamp": i, "close": 100 + i} for i in range(100)])

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().values.any():
            print("[⚠️ compute_features 결과 이상 → dummy로 대체]")
            df_feat = pd.DataFrame(np.random.normal(0, 1, size=(100, input_size)), columns=[f"f{i}" for i in range(input_size)])

        try:
            dummy_X = torch.tensor(np.random.rand(10, 20, input_size), dtype=torch.float32).to(DEVICE)
            dummy_y = torch.randint(0, 2, (10,), dtype=torch.long).to(DEVICE)
            importances = compute_feature_importance(get_model("lstm", input_size=input_size, output_size=2).to(DEVICE), dummy_X, dummy_y, [c for c in df_feat.columns if c not in ["timestamp", "strategy"]], method="baseline")
            df_feat = drop_low_importance_features(df_feat, importances, threshold=0.01, min_features=5)
        except: pass

        for window in find_best_windows(symbol, strategy) or [20]:
            print(f"▶️ window={window} → dataset 생성 시작")
            X_y = create_dataset(df_feat.to_dict(orient="records"), window=window, strategy=strategy, input_size=input_size)
            if not X_y or not isinstance(X_y, tuple) or len(X_y) != 2:
                print(f"[❌ create_dataset unpack 실패] → window={window}")
                log_training_result(symbol, strategy, f"학습실패:dataset_unpack실패_window{window}", 0.0, 0.0, 0.0)
                continue

            X_raw, y_raw = X_y
            if X_raw is None or y_raw is None or len(X_raw) == 0:
                print(f"[❌ dataset 생성 실패] → window={window}")
                log_training_result(symbol, strategy, f"학습실패:dataset없음_window{window}", 0.0, 0.0, 0.0)
                continue

            fail_X, fail_y = load_training_prediction_data(symbol, strategy, input_size=input_size, window=window)
            if fail_X is not None and len(fail_X) > 0:
                X_raw = np.concatenate([X_raw, fail_X], axis=0)
                y_raw = np.concatenate([y_raw, fail_y], axis=0)

            val_len = max(5, int(len(X_raw) * 0.2))
            if len(X_raw) <= val_len:
                val_indices = np.random.choice(len(X_raw), val_len, replace=True)
                X_val, y_val, X_train, y_train = X_raw[val_indices], y_raw[val_indices], X_raw, y_raw
            else:
                X_train, y_train, X_val, y_val = X_raw[:-val_len], y_raw[:-val_len], X_raw[-val_len:], y_raw[-val_len:]

            for gid in [group_id] if group_id is not None else range(len(class_groups_list)):
                group_classes = class_groups_list[gid]
                print(f"▶️ 학습 group{gid}: 클래스 수 {len(group_classes)}")

                if not group_classes:
                    log_training_result(symbol, strategy, f"학습실패:빈그룹_group{gid}_window{window}", 0.0, 0.0, 0.0)
                    continue

                tm = np.isin(y_train, group_classes); vm = np.isin(y_val, group_classes)
                X_train_group, y_train_group = X_train[tm], y_train[tm]
                X_val_group, y_val_group = X_val[vm], y_val[vm]

                print(f"▶️ group{gid} → train:{len(y_train_group)}, val:{len(y_val_group)}")

                if len(y_train_group) < 2:
                    print(f"[❌ train 부족] group{gid}")
                    log_training_result(symbol, strategy, f"학습실패:데이터부족_group{gid}_window{window}", 0.0, 0.0, 0.0)
                    continue
                if len(y_val_group) == 0:
                    print(f"[❌ val 없음] group{gid}")
                    log_training_result(symbol, strategy, f"학습실패:val없음_group{gid}_window{window}", 0.0, 0.0, 0.0)
                    continue

                try:
                    y_train_group = np.array([group_classes.index(y) for y in y_train_group])
                    y_val_group = np.array([group_classes.index(y) for y in y_val_group])
                except Exception as e:
                    log_training_result(symbol, strategy, f"학습실패:label불일치_group{gid}_window{window}", 0.0, 0.0, 0.0)
                    continue

                X_train_group, y_train_group = balance_classes(X_train_group, y_train_group, min_count=20, num_classes=len(group_classes))

                for model_type in ["lstm", "cnn_lstm", "transformer"]:
                    try:
                        model = get_model(model_type, input_size=input_size, output_size=len(group_classes)).to(DEVICE).train()
                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                        lossfn = FocalLoss()
                        to_tensor = lambda x: torch.tensor(x[:, -1, :], dtype=torch.float32)
                        Xtt, ytt = to_tensor(X_train_group), torch.tensor(y_train_group, dtype=torch.long)
                        Xvt, yvt = to_tensor(X_val_group), torch.tensor(y_val_group, dtype=torch.long)
                        train_loader = DataLoader(TensorDataset(Xtt, ytt), batch_size=32, shuffle=True)
                        val_loader = DataLoader(TensorDataset(Xvt, yvt), batch_size=32)

                        for _ in range(max_epochs):
                            for xb, yb in train_loader:
                                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                                loss = lossfn(model(xb), yb)
                                if torch.isfinite(loss): optimizer.zero_grad(); loss.backward(); optimizer.step()

                        maml_train_entry(model, train_loader, val_loader, inner_lr=0.01, outer_lr=0.001, inner_steps=1)
                        model.eval()
                        with torch.no_grad():
                            val_preds = torch.argmax(model(Xvt.to(DEVICE)), dim=1)
                            val_acc = (val_preds == yvt.to(DEVICE)).float().mean().item()

                        model_name = f"{model_type}_AdamW_FocalLoss_lr1e-4_bs=32_hs=64_dr=0.3_group{gid}_window{window}"
                        model_path = f"/persistent/models/{symbol}_{strategy}_{model_name}.pt"
                        torch.save(model.state_dict(), model_path)
                        print(f"[✅ 저장됨] {model_path}")

                        meta = {
                            "symbol": symbol,
                            "strategy": strategy,
                            "model": model_type,
                            "group_id": gid,
                            "window": window,
                            "input_size": input_size,
                            "output_size": len(group_classes),
                            "model_name": model_name,
                            "timestamp": now_kst().isoformat()
                        }

                        with open(model_path.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        log_training_result(symbol, strategy, model_name, acc=val_acc, f1=0.0, loss=loss.item())
                        trained_any = True

                        del model, optimizer, lossfn, train_loader, val_loader
                        torch.cuda.empty_cache(); gc.collect()
                    except Exception as inner_e:
                        reason = f"{type(inner_e).__name__}: {inner_e}"
                        log_training_result(symbol, strategy, f"학습실패:{reason}", 0.0, 0.0, 0.0)
                        print(f"[❌ 내부 예외] group{gid} window{window} → {reason}")

        if not trained_any:
            print(f"[❌ 모델 전부 실패] 저장된 모델 없음 → {symbol}-{strategy}")
            log_training_result(symbol, strategy, f"학습실패:전그룹학습불가", 0.0, 0.0, 0.0)

    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        log_training_result(symbol, strategy, f"학습실패:전체예외:{reason}", 0.0, 0.0, 0.0)
        print(f"[❌ 전체 예외] {symbol}-{strategy}: {reason}")


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
    - 심볼 하나당: 클래스그룹 전체 → 전략 순서로 학습
    - 모든 그룹/전략 학습 완료 후 다음 심볼로 넘어감
    - 예측 실행은 하지 않음 (외부 recommend에서 호출)
    - meta.json 일괄 보정 수행
    """
    global training_in_progress
    from telegram_bot import send_message
    import maintenance_fix_meta
    from config import get_class_groups
    import time

    strategies = ["단기", "중기", "장기"]
    class_groups = get_class_groups()
    group_ids = list(range(len(class_groups)))

    print(f"🚀 [train_models] 심볼 학습 시작: {symbol_list}")

    for symbol in symbol_list:
        print(f"\n🔁 [심볼 시작] {symbol}")

        for group_id in group_ids:
            print(f"▶ 그룹 {group_id} 학습 시작")

            for strategy in strategies:
                if training_in_progress.get(strategy, False):
                    print(f"⚠️ 중복 실행 방지: {strategy}")
                    continue

                training_in_progress[strategy] = True
                try:
                    train_one_model(symbol, strategy, group_id=group_id)
                except Exception as e:
                    print(f"[❌ 학습 실패] {symbol}-{strategy}-group{group_id} → {e}")
                finally:
                    training_in_progress[strategy] = False
                    print(f"✅ {symbol}-{strategy}-group{group_id} 학습 완료")
                    time.sleep(2)

    # ✅ 모든 학습 후 메타 보정
    try:
        maintenance_fix_meta.fix_all_meta_json()
        print(f"✅ meta 보정 완료: {symbol_list}")
    except Exception as e:
        print(f"[⚠️ meta 보정 실패] {e}")

    # ✅ 실패학습 자동 실행 추가
    try:
        import failure_trainer
        failure_trainer.run_failure_training()
    except Exception as e:
        print(f"[❌ 실패학습 루프 예외] {e}")

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
    """
    ✅ 심볼 → 전략 순서로 순차 학습되도록 개선
    ✅ 심볼별 전략별 클래스 전체 그룹 학습 완료 후 다음 심볼로 이동
    """
    import time
    import maintenance_fix_meta
    from data.utils import SYMBOL_GROUPS, _kline_cache, _feature_cache
    from train import train_one_model

    group_count = len(SYMBOL_GROUPS)
    print(f"🚀 전체 {group_count}개 그룹 학습 루프 시작")

    loop_count = 0
    while True:
        loop_count += 1
        print(f"\n🔄 그룹 학습 루프 #{loop_count} 시작")

        for idx, group in enumerate(SYMBOL_GROUPS):
            print(f"\n🚀 [그룹 {idx}/{group_count}] 학습 시작 | 심볼: {group}")

            _kline_cache.clear()
            _feature_cache.clear()
            print("[✅ cache cleared] _kline_cache, _feature_cache")

            try:
                # ✅ 각 심볼에 대해 전략 순차 학습 (→ 클래스 그룹 전체 학습)
                for symbol in group:
                    for strategy in ["단기", "중기", "장기"]:
                        try:
                            train_one_model(symbol, strategy, group_id=None)  # ✅ 핵심 수정
                            print(f"[✅ 학습 완료] {symbol}-{strategy}")
                        except Exception as e:
                            print(f"[❌ 학습 실패] {symbol}-{strategy} → {e}")

                # ✅ 메타 정보 보정
                maintenance_fix_meta.fix_all_meta_json()
                print(f"[✅ meta 보정 완료] 그룹 {idx}")

                # ✅ 학습 후 예측까지 자동 수행
                from recommend import main
                for symbol in group:
                    for strategy in ["단기", "중기", "장기"]:
                        try:
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
