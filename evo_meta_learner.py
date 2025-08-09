# === evo_meta_learner.py 최종본 ===
import os
import json
import math
import time
import traceback
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# 기본 경로/상수
MODEL_DIR = "/persistent/models"
MODEL_PATH = os.path.join(MODEL_DIR, "evo_meta_learner.pt")
META_PATH = MODEL_PATH.replace(".pt", ".meta.json")
PRED_LOG = "/persistent/prediction_log.csv"   # 로그 경로 통일 기준
DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────────────────────
class EvoMetaModel(nn.Module):
    """
    간단한 MLP 기반 메타러너:
      입력: feature_vector (마지막 timestep, FEATURE_INPUT_SIZE)
      출력: num_classes 로지츠
    """
    def __init__(self, input_size: int, num_classes: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, input_size)
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────
def _read_meta() -> Tuple[int, int]:
    """
    META_PATH에서 (input_size, num_classes) 읽기.
    없으면 보수적 기본값 (input_size=64, num_classes=3) 반환.
    """
    default = (64, 3)
    if not os.path.exists(META_PATH):
        return default
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        input_size = int(meta.get("input_size", default[0]))
        num_classes = int(meta.get("num_classes", default[1]))
        return input_size, num_classes
    except Exception:
        return default


def _write_meta(input_size: int, num_classes: int) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    meta = {
        "input_size": int(input_size),
        "num_classes": int(num_classes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "evo_meta_learner",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _parse_feature_vector(val) -> Optional[np.ndarray]:
    """
    prediction_log.csv 의 feature_vector 컬럼을 numpy array 로 파싱.
    - JSON 문자열(list) 또는 리스트 그대로 허용.
    - 유효하지 않으면 None.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        if isinstance(val, str):
            vec = json.loads(val)
        else:
            vec = val
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        # 2D로 들어온 경우 마지막 축이 feature 라고 가정 → 마지막 timestep 사용
        if arr.ndim == 2:
            return arr[-1]
        return None
    except Exception:
        return None


def _load_training_rows(input_size: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    prediction_log.csv에서 feature_vector 와 라벨(label 또는 predicted_class)을 읽어 학습 데이터 구성.
    """
    if not os.path.exists(PRED_LOG):
        return None, None, 0
    try:
        df = pd.read_csv(PRED_LOG, encoding="utf-8-sig")
    except Exception:
        return None, None, 0

    need = {"feature_vector"}
    if not need.issubset(df.columns):
        return None, None, 0

    # 라벨 우선순위: label -> predicted_class
    label_col = "label" if "label" in df.columns else ("predicted_class" if "predicted_class" in df.columns else None)
    if label_col is None:
        return None, None, 0

    xs, ys = [], []
    for _, row in df.iterrows():
        fv = _parse_feature_vector(row.get("feature_vector"))
        if fv is None:
            continue
        if len(fv) != input_size:
            # 패딩/자르기
            if len(fv) < input_size:
                fv = np.pad(fv, (0, input_size - len(fv)), mode="constant")
            else:
                fv = fv[:input_size]

        y_raw = row.get(label_col)
        try:
            y = int(y_raw)
        except Exception:
            continue

        if y < 0:
            continue

        xs.append(fv.astype(np.float32))
        ys.append(y)

    if not xs:
        return None, None, 0

    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    num_classes_observed = int(np.max(y) + 1)
    return X, y, num_classes_observed


# ──────────────────────────────────────────────────────────────
# 학습 / 루프
# ──────────────────────────────────────────────────────────────
def train_evo_meta(input_size: Optional[int] = None, num_classes_hint: Optional[int] = None, epochs: int = 10) -> Optional[str]:
    """
    prediction_log.csv 기반으로 간단 학습을 수행.
    - 데이터 부족 시 None 반환(조용히 스킵)
    - 성공 시 모델 저장 후 모델 경로 반환
    """
    try:
        if input_size is None:
            # 기존 메타가 있으면 그 스펙 따름
            input_size, _nc = _read_meta()

        X, y, observed_classes = _load_training_rows(input_size)
        if X is None or y is None or len(X) < 50:
            print(f"[evo_meta] 학습 데이터 부족 → 스킵 (rows={0 if X is None else len(X)})")
            return None

        num_classes = num_classes_hint or observed_classes or 3
        num_classes = max(2, int(num_classes))

        model = EvoMetaModel(input_size=input_size, num_classes=num_classes).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        lossfn = nn.CrossEntropyLoss()

        # 간단한 train/val split
        n = len(X)
        val_n = max(1, int(n * 0.2))
        X_train, y_train = X[:-val_n], y[:-val_n]
        X_val, y_val = X[-val_n:], y[-val_n:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=DEVICE)

        for ep in range(epochs):
            model.train()
            logits = model(X_train_t)
            loss = lossfn(logits, y_train_t)
            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                model.eval()
                val_logits = model(X_val_t)
                val_pred = torch.argmax(val_logits, dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
            print(f"[evo_meta][{ep+1}/{epochs}] loss={loss.item():.4f} val_acc={val_acc:.3f}")

        # 저장
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        _write_meta(input_size=input_size, num_classes=num_classes)
        print(f"[evo_meta] 학습 완료 → {MODEL_PATH} (num_classes={num_classes})")
        return MODEL_PATH

    except Exception as e:
        print(f"[❌ 진화형 메타러너 학습 실패] {e}")
        traceback.print_exc()
        return None


def train_evo_meta_loop(interval_minutes: int = 60, max_once: bool = True):
    """
    주기적으로 학습을 갱신. (train_symbol_group_loop에서 호출)
    - max_once=True면 1회만 실행 후 반환.
    """
    try:
        train_evo_meta()  # 입력 크기는 META나 로그 기반으로 자동 추정
    except Exception as e:
        print(f"[⚠️ 진화형 메타러너 학습 루프 예외] {e}")

    if max_once:
        return

    while True:
        time.sleep(interval_minutes * 60)
        try:
            train_evo_meta()
        except Exception as e:
            print(f"[⚠️ 진화형 메타러너 재학습 예외] {e}")


# ──────────────────────────────────────────────────────────────
# 예측
# ──────────────────────────────────────────────────────────────
def predict_evo_meta(X_new: torch.Tensor, input_size: int) -> Optional[int]:
    """
    X_new: (B, input_size) 또는 (1, input_size) 텐서
    - 모델 파일 없으면 None
    - 메타(.meta.json)에서 num_classes 읽어 모델 생성
    """
    if not os.path.exists(MODEL_PATH):
        print("[❌ evo_meta_learner] 모델 없음")
        return None

    # 메타에서 스펙 확인
    meta_input_size, num_classes = _read_meta()
    if meta_input_size != input_size:
        # 입력 크기 불일치 시, 가능한 한 맞춰서 시도 (패딩/자르기)
        input_size = meta_input_size

    try:
        model = EvoMetaModel(input_size=input_size, num_classes=num_classes).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as e:
        print(f"[❌ evo_meta_learner 로드 실패] {e}")
        return None

    try:
        if not isinstance(X_new, torch.Tensor):
            X_new = torch.tensor(np.asarray(X_new, dtype=np.float32))
        if X_new.ndim == 1:
            X_new = X_new.unsqueeze(0)
        # 패딩/자르기로 입력 크기 맞춤
        if X_new.shape[1] != input_size:
            if X_new.shape[1] < input_size:
                pad = input_size - X_new.shape[1]
                X_new = torch.cat([X_new, torch.zeros((X_new.shape[0], pad), dtype=X_new.dtype)], dim=1)
            else:
                X_new = X_new[:, :input_size]

        with torch.no_grad():
            logits = model(X_new.to(DEVICE))
            pred = int(torch.argmax(logits, dim=1).item())
        return pred
    except Exception as e:
        print(f"[⚠️ evo_meta_learner 예측 예외] {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 스크립트 실행 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 간단 동작 점검
    # 1) 학습 시도 (로그가 충분하지 않으면 자동 스킵)
    train_evo_meta()

    # 2) 예측 시도 (더미 입력)
    inp_size, _nc = _read_meta()
    x = torch.randn(1, inp_size)
    pred = predict_evo_meta(x, input_size=inp_size)
    print("[evo_meta predict]", pred)
