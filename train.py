import os, json, time, traceback
from datetime import datetime
import pytz
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from data.utils import SYMBOLS, get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from feature_importance import compute_feature_importance, save_feature_importance
from failure_db import load_existing_failure_hashes, insert_failure_record, ensure_failure_db
from logger import log_training_result
from window_optimizer import find_best_window
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE, get_class_groups, get_class_ranges, set_NUM_CLASSES
from data_augmentation import balance_classes

# ✅ SSL 프리트레인 (파일명: ssl_pretrain.py)
from ssl_pretrain import masked_reconstruction

# ✅ 진화형 메타 러너: 루프 호출로 통일
from evo_meta_learner import train_evo_meta_loop

NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "/persistent/models"
os.makedirs(MODEL_DIR, exist_ok=True)

now_kst = lambda: datetime.now(pytz.timezone("Asia/Seoul"))
training_in_progress = {"단기": False, "중기": False, "장기": False}

# --------------------------------------------------
# 유틸
# --------------------------------------------------
def get_feature_hash_from_tensor(x, use_full=False, precision=3):
    import hashlib
    if x.ndim != 2 or x.shape[0] == 0:
        return "invalid"
    try:
        flat = x.flatten() if use_full else x[-1]
        rounded = [round(float(val), precision) for val in flat]
        return hashlib.sha1(",".join(map(str, rounded)).encode()).hexdigest()
    except Exception:
        return "invalid"

# --------------------------------------------------
# 단일 (symbol, strategy, group_id) 모델 학습
# --------------------------------------------------
def train_one_model(symbol, strategy, group_id=None, max_epochs=20):
    """
    - SSL 사전학습 실행 (실패해도 계속)
    - 가격/피처 로드 → 라벨링 → 윈도우 시퀀스 구성
    - 필요 시 클래스 밸런싱
    - [lstm, cnn_lstm, transformer] 각각 학습/평가/저장
    - 메타파일(.meta.json) 동시 저장
    """
    try:
        print(f"✅ [train_one_model 시작] {symbol}-{strategy}-group{group_id}")
        ensure_failure_db()

        # 0) SSL 프리트레인 (실패해도 통과)
        try:
            masked_reconstruction(symbol, strategy, FEATURE_INPUT_SIZE)
        except Exception as e:
            print(f"[⚠️ SSL 사전학습 실패] {e}")

        # 1) 데이터 로드
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or df.empty:
            print(f"[⏩ 스킵] {symbol}-{strategy}-group{group_id} → 데이터 없음")
            log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                                loss=0.0, note="데이터 없음", status="skipped")
            insert_failure_record({
                "symbol": symbol, "strategy": strategy, "model": "all",
                "predicted_class": -1, "success": False, "rate": "", "reason": "데이터 없음"
            }, feature_vector=[])
            return

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.empty or feat.isnull().any().any():
            print(f"[⏩ 스킵] {symbol}-{strategy}-group{group_id} → 피처 없음/NaN")
            log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                                loss=0.0, note="피처 없음", status="skipped")
            insert_failure_record({
                "symbol": symbol, "strategy": strategy, "model": "all",
                "predicted_class": -1, "success": False, "rate": "", "reason": "피처 없음"
            }, feature_vector=[])
            return

        features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        feat_scaled = MinMaxScaler().fit_transform(features_only)

        # 2) 클래스 경계/라벨링
        try:
            class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
        except Exception as e:
            print(f"[❌ 클래스 범위 계산 실패] {e}")
            log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                                loss=0.0, note="클래스 계산 실패", status="failed")
            insert_failure_record({
                "symbol": symbol, "strategy": strategy, "model": "all",
                "predicted_class": -1, "success": False, "rate": "", "reason": "클래스 계산 실패"
            }, feature_vector=[])
            return

        num_classes = len(class_ranges)
        set_NUM_CLASSES(num_classes)

        returns = df["close"].pct_change().fillna(0).values
        labels = []
        for r in returns:
            idx = 0
            for i, (lo, hi) in enumerate(class_ranges):
                if lo <= r <= hi:
                    idx = i
                    break
            labels.append(idx)

        # 3) 시퀀스 생성
        window = 60
        X, y = [], []
        for i in range(len(feat_scaled) - window):
            X.append(feat_scaled[i:i+window])
            y.append(labels[i + window] if i + window < len(labels) else 0)
        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

        if len(X) < 10:
            print(f"[⏩ 스킵] {symbol}-{strategy}-group{group_id} → 샘플 부족({len(X)})")
            log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                                loss=0.0, note="최종 샘플 부족", status="skipped")
            insert_failure_record({
                "symbol": symbol, "strategy": strategy, "model": "all",
                "predicted_class": -1, "success": False, "rate": "", "reason": "최종 샘플 부족"
            }, feature_vector=[])
            return

        # 4) 밸런싱(필요 시)
        try:
            if len(X) < 50:
                X, y = balance_classes(X, y, num_classes=num_classes)
                print(f"[✅ 증강/밸런싱 완료] 총 샘플: {len(X)}")
        except Exception as e:
            print(f"[⚠️ 밸런싱 실패] {e}")

        # 5) 학습/평가/저장 (모델 3종)
        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            print(f"[🧠 학습 시작] {model_type} | {symbol}-{strategy}-group{group_id}")
            model = get_model(model_type, input_size=FEATURE_INPUT_SIZE, output_size=num_classes).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            # train/val split
            val_len = max(1, int(len(X) * 0.2))
            if len(X) - val_len < 1:
                val_len = len(X) - 1
            train_X, val_X = X[:-val_len], X[-val_len:]
            train_y, val_y = y[:-val_len], y[-val_len:]

            train_loader = DataLoader(TensorDataset(
                torch.tensor(train_X), torch.tensor(train_y)), batch_size=64, shuffle=True)
            val_loader = DataLoader(TensorDataset(
                torch.tensor(val_X), torch.tensor(val_y)), batch_size=64)

            total_loss = 0.0
            for epoch in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    if not torch.isfinite(loss):
                        continue
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total_loss += loss.item()
                if (epoch + 1) % 5 == 0 or epoch == max_epochs - 1:
                    print(f"[📈 {model_type}] Epoch {epoch+1}/{max_epochs} | loss={loss.item():.4f}")

            # 평가
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = torch.argmax(model(xb.to(DEVICE)), dim=1).cpu().numpy()
                    all_preds.extend(preds); all_labels.extend(yb.numpy())
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")
            print(f"[🎯 {model_type}] acc={acc:.4f}, f1={f1:.4f}")

            # 저장
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_name = f"{symbol}_{strategy}_{model_type}_group{group_id}_cls{num_classes}.pt"
            model_path = os.path.join(MODEL_DIR, model_name)
            torch.save(model.state_dict(), model_path)

            meta = {
                "symbol": symbol, "strategy": strategy, "model": model_type,
                "group_id": int(group_id) if group_id is not None else 0,
                "num_classes": int(num_classes),
                "input_size": int(FEATURE_INPUT_SIZE),
                "timestamp": now_kst().isoformat(),
                "model_name": model_name
            }
            with open(model_path.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # ✅ log_training_result 호출 형식 통일
            log_training_result(
                symbol=symbol,
                strategy=strategy,
                model=model_name,
                accuracy=float(acc),
                f1=float(f1),
                loss=float(total_loss),
                note="train_one_model",
                source_exchange="BYBIT",
                status="success",
            )

        print(f"[✅ train_one_model 완료] {symbol}-{strategy}-group{group_id}")

    except Exception as e:
        print(f"[❌ train_one_model 실패] {symbol}-{strategy}-group{group_id} → {e}")
        traceback.print_exc()
        log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                            loss=0.0, note=str(e), status="failed")
        insert_failure_record({
            "symbol": symbol, "strategy": strategy, "model": "all",
            "predicted_class": -1, "success": False, "rate": "", "reason": str(e)
        }, feature_vector=[])

# --------------------------------------------------
# 전체 학습 루틴
# --------------------------------------------------
def train_models(symbol_list):
    """
    - 심볼 × (단기/중기/장기) × 전 그룹 학습
    - 메타 보정/실패학습/진화형 메타 루프 호출
    """
    strategies = ["단기", "중기", "장기"]
    print(f"🚀 [train_models] 심볼 학습 시작: {symbol_list}")

    for symbol in symbol_list:
        print(f"\n🔁 [심볼] {symbol}")
        for strategy in strategies:
            print(f"▶ {symbol}-{strategy} 전체 그룹 학습")
            # group_id 목록 계산
            try:
                class_ranges = get_class_ranges(symbol=symbol, strategy=strategy)
                if not class_ranges:
                    raise ValueError("빈 클래스 경계")
                num_classes = len(class_ranges)
                groups = get_class_groups(num_classes=num_classes)  # 리스트들의 리스트
                max_gid = len(groups) - 1
            except Exception as e:
                print(f"[❌ 클래스 경계 계산 실패] {symbol}-{strategy}: {e}")
                log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                                    loss=0.0, note=f"클래스 계산 실패: {e}", status="failed")
                continue

            for gid in range(max_gid + 1):
                train_one_model(symbol, strategy, group_id=gid)
                time.sleep(0.5)

    # 메타 보정(있다면)
    try:
        import maintenance_fix_meta
        maintenance_fix_meta.fix_all_meta_json()
    except Exception as e:
        print(f"[⚠️ meta 보정 실패] {e}")

    # 실패학습 루프
    try:
        import failure_trainer
        failure_trainer.run_failure_training()
    except Exception as e:
        print(f"[⚠️ 실패학습 루프 예외] {e}")

    # ✅ 진화형 메타러너 학습 루프 호출(단발)
    try:
        train_evo_meta_loop()
    except Exception as e:
        print(f"[⚠️ 진화형 메타러너 학습 실패] {e}")

    print("✅ train_models 완료")

def train_all_models():
    """SYMBOLS 전체 반복 학습 (간단 버전)"""
    train_models(SYMBOLS)

# --------------------------------------------------
# 개별 전략 무한 루프 (옵션)
# --------------------------------------------------
def train_model_loop(strategy):
    global training_in_progress
    if training_in_progress.get(strategy, False):
        print(f"⚠️ 중복 실행 방지: {strategy}")
        return
    training_in_progress[strategy] = True
    print(f"🚀 {strategy} 무한 학습 루프 시작")

    try:
        for symbol in SYMBOLS:
            train_one_model(symbol, strategy, group_id=0)
    finally:
        training_in_progress[strategy] = False
        print(f"✅ {strategy} 루프 종료")

if __name__ == "__main__":
    # 예시: 전 심볼 학습
    train_all_models()
