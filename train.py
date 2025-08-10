import os, json, time, traceback
from datetime import datetime
import pytz
import numpy as np
import pandas as pd  # ⬅ 추가: 미래 수익률 계산에 필요
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from data.utils import SYMBOLS, get_kline_by_strategy, compute_features, create_dataset
from model.base_model import get_model
from model_weight_loader import get_model_weight
from feature_importance import compute_feature_importance, save_feature_importance
from failure_db import insert_failure_record, ensure_failure_db
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


def _log_skip(symbol, strategy, reason):
    log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                        loss=0.0, note=reason, status="skipped")
    insert_failure_record({
        "symbol": symbol, "strategy": strategy, "model": "all",
        "predicted_class": -1, "success": False, "rate": "", "reason": reason
    }, feature_vector=[])


def _log_fail(symbol, strategy, reason):
    log_training_result(symbol, strategy, model="all", accuracy=0.0, f1=0.0,
                        loss=0.0, note=reason, status="failed")
    insert_failure_record({
        "symbol": symbol, "strategy": strategy, "model": "all",
        "predicted_class": -1, "success": False, "rate": "", "reason": reason
    }, feature_vector=[])


def _strategy_horizon_hours(strategy: str) -> int:
    """전략별 평가 구간(시간). predict/evaluate와 합치기 위해 동일 기준 사용."""
    return {"단기": 4, "중기": 24, "장기": 168}.get(strategy, 24)


def _future_returns_by_timestamp(df: pd.DataFrame, horizon_hours: int) -> np.ndarray:
    """
    각 시점 t에 대해 [t, t+horizon] 구간의 'max(high)'를 사용하여
    미래 최대 수익률을 계산 ( (max_high - close_t) / close_t ).
    길이는 df와 동일하게 맞춤. 마지막 구간은 데이터 부족 시 0으로 채움.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return np.zeros(len(df), dtype=np.float32)

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    close = df["close"].astype(float).values
    high = (df["high"] if "high" in df.columns else df["close"]).astype(float).values

    # 타임존 정규화
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")

    out = np.zeros(len(df), dtype=np.float32)
    horizon = pd.Timedelta(hours=horizon_hours)

    # 슬라이딩 윈도우 방식(간단 구현)
    j_start = 0
    for i in range(len(df)):
        t0 = ts.iloc[i]
        t1 = t0 + horizon
        # j 커서 앞으로 이동
        j = max(j_start, i)
        max_h = high[i]
        while j < len(df) and ts.iloc[j] <= t1:
            if high[j] > max_h:
                max_h = high[j]
            j += 1
        j_start = max(j_start, i)  # 커서 관리(최적화 미세)
        base = close[i] if close[i] > 0 else (close[i] + 1e-6)
        out[i] = float((max_h - base) / (base + 1e-12))
    return out.astype(np.float32)

# --------------------------------------------------
# 단일 (symbol, strategy, group_id) 모델 학습
# --------------------------------------------------
def train_one_model(symbol, strategy, group_id=None, max_epochs=20):
    """
    - SSL 사전학습 실행 (실패해도 계속)
    - 가격/피처 로드 → (미래 수익률 기반) 라벨링 → 윈도우 시퀀스 구성
    - find_best_window()로 동적 윈도우 선택
    - 필요 시 클래스 밸런싱 및 마지막 안전 보강(fallback)으로 스킵 최소화
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
            _log_skip(symbol, strategy, "데이터 없음")
            return

        feat = compute_features(symbol, df, strategy)
        if feat is None or feat.empty or feat.isnull().any().any():
            print(f"[⏩ 스킵] {symbol}-{strategy}-group{group_id} → 피처 없음/NaN")
            _log_skip(symbol, strategy, "피처 없음")
            return

        # 2) 클래스 경계/라벨링 (✅ 미래 수익률 기반)
        try:
            class_ranges = get_class_ranges(symbol=symbol, strategy=strategy, group_id=group_id)
        except Exception as e:
            print(f"[❌ 클래스 범위 계산 실패] {e}")
            _log_fail(symbol, strategy, "클래스 계산 실패")
            return

        num_classes = len(class_ranges)
        set_NUM_CLASSES(num_classes)

        # 미래 수익률
        horizon_hours = _strategy_horizon_hours(strategy)
        future_gains = _future_returns_by_timestamp(df, horizon_hours=horizon_hours)

        # 클래스 매핑
        labels = []
        for r in future_gains:
            idx = 0
            for i, (lo, hi) in enumerate(class_ranges):
                if lo <= r <= hi:
                    idx = i
                    break
            labels.append(idx)
        labels = np.array(labels, dtype=np.int64)

        # 3) 동적 윈도우 선택
        try:
            best_window = find_best_window(symbol, strategy, window_list=[10, 20, 30, 40, 60])
        except Exception as e:
            print(f"[⚠️ find_best_window 실패] {e}")
            best_window = 60
        window = int(max(5, best_window))
        print(f"[🔧 선택된 WINDOW] {symbol}-{strategy} → {window}")

        # 4) 시퀀스 생성 (기본 경로)
        features_only = feat.drop(columns=["timestamp", "strategy"], errors="ignore")
        feat_scaled = MinMaxScaler().fit_transform(features_only)

        X, y = [], []
        # 윈도우 끝 시점의 '미래 수익률 라벨'을 정답으로 사용
        for i in range(len(feat_scaled) - window):
            X.append(feat_scaled[i:i+window])
            y_idx = i + window - 1  # 윈도우 끝 시점 인덱스
            y.append(labels[y_idx] if 0 <= y_idx < len(labels) else 0)
        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
        print(f"[📊 초기 시퀀스] {symbol}-{strategy} → {len(y)}건")

        # 4-1) 마지막 안전 보강: 샘플 너무 적으면 create_dataset fallback
        if len(X) < 20:
            print("[ℹ️ 안전 보강: create_dataset fallback 사용]")
            feat_records = feat.to_dict(orient="records")
            try:
                # create_dataset은 미래 수익률 기반(look-ahead) 로직을 내장
                res = create_dataset(feat_records, window=window, strategy=strategy, input_size=FEATURE_INPUT_SIZE)
                if isinstance(res, tuple) and len(res) >= 2:
                    X_fb, y_fb = res[0], res[1]
                else:
                    X_fb, y_fb = res
                if isinstance(X_fb, np.ndarray) and len(X_fb) > 0:
                    X, y = X_fb.astype(np.float32), y_fb.astype(np.int64)
                    print(f"[✅ fallback 적용] 최종 샘플: {len(y)}")
            except Exception as e:
                print(f"[⚠️ fallback 실패] {e}")

        if len(X) < 10:
            print(f"[⏩ 스킵] {symbol}-{strategy}-group{group_id} → 최종 샘플 부족({len(X)})")
            _log_skip(symbol, strategy, "최종 샘플 부족")
            return

        # 5) 밸런싱(필요 시)
        try:
            if len(X) < 200:  # 규모가 작을 때만 과도한 증강 방지
                X, y = balance_classes(X, y, num_classes=num_classes)
                print(f"[✅ 증강/밸런싱 완료] 총 샘플: {len(X)}")
        except Exception as e:
            print(f"[⚠️ 밸런싱 실패] {e}")

        # 6) 학습/평가/저장 (모델 3종)
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

            log_training_result(
                symbol=symbol,
                strategy=strategy,
                model=model_name,
                accuracy=float(acc),
                f1=float(f1),
                loss=float(total_loss),
                note=f"train_one_model(window={window})",
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
                _log_fail(symbol, strategy, f"클래스 계산 실패: {e}")
                continue

            for gid in range(max_gid + 1):
                train_one_model(symbol, strategy, group_id=gid)
                time.sleep(0.5)

    # 메타 보정
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
