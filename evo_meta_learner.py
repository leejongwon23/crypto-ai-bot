import os, json, ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "/persistent/models/evo_meta_learner.pt"
META_PATH  = MODEL_PATH.replace(".pt", ".meta.json")

class EvoMetaModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def _safe_lit(x, default):
    try:
        if x is None or str(x).strip() == "":
            return default
        return ast.literal_eval(str(x))
    except Exception:
        return default

def prepare_evo_meta_dataset(path="/persistent/wrong_predictions.csv", min_samples=50):
    """
    실패/오답 로그를 기반으로 '전략 선택' 학습용 피처 구성.
    - softmax, expected_returns, model_predictions 컬럼 사용(없으면 스킵)
    - label, best_strategy 사용(없으면 기본값)
    """
    if not os.path.exists(path):
        print(f"[❌ prepare_evo_meta_dataset] 파일 없음: {path}")
        return None, None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)

    if len(df) < min_samples:
        print(f"[❌ prepare_evo_meta_dataset] 샘플 부족: {len(df)}개")
        return None, None

    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            sm = _safe_lit(row.get("softmax"), [])
            er = _safe_lit(row.get("expected_returns"), [0, 0, 0])
            mp = _safe_lit(row.get("model_predictions"), [0, 0, 0])
            if not sm or len(sm) != 3:
                continue
            label = int(float(row.get("label", -1))) if pd.notnull(row.get("label")) else -1
            # 3전략 기준 간단 피처: [softmax_i, expected_return_i, hit_flag_i] * 3
            feats = []
            for i in range(3):
                hit = 1 if (i < len(mp) and mp[i] == label) else 0
                feats.extend([float(sm[i]), float(er[i] if i < len(er) else 0.0), hit])
            X_list.append(feats)
            y_list.append(int(float(row.get("best_strategy", 0))))  # 0/1/2
        except Exception:
            continue

    if not X_list or not y_list:
        print("[❌ prepare_evo_meta_dataset] 유효 샘플 부족")
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[✅ prepare_evo_meta_dataset] X:{X.shape}, y:{y.shape}")
    return X, y

def train_evo_meta(X, y, input_size, output_size=3, epochs=10, batch_size=32, lr=1e-3, task="strategy"):
    """
    task: "strategy"(기본) 또는 "class" — 메타 JSON에 기록되어 추론 시 가드로 사용.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvoMetaModel(input_size, output_size=output_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[evo_meta_learner] 학습 시작 → 샘플:{len(dataset)}, output_size:{output_size}, input:{input_size}, task={task}")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss.item())
        print(f"[evo_meta_learner] Epoch {epoch+1}/{epochs} → Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    meta_info = {
        "task": str(task),               # 👈 중요: "strategy" 또는 "class"
        "input_size": int(input_size),
        "output_size": int(output_size),
        "version": 1
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    print(f"[✅ evo_meta_learner] 학습 완료 → {MODEL_PATH} (task={task}, output_size={output_size})")
    return model

def train_evo_meta_loop(min_samples=50, auto_trigger=False, task="strategy"):
    """
    기본은 '전략 선택' 학습. (task='strategy')
    만약 클래스 예측용 데이터셋이 따로 마련되면 task='class'로 호출해도 됨.
    """
    X, y = prepare_evo_meta_dataset(min_samples=min_samples) if task == "strategy" else (None, None)
    if X is None or y is None:
        if auto_trigger:
            print("[⏭️ evo_meta_learner] 데이터 부족 → 자동 학습 스킵")
        return
    input_size = X.shape[1]
    train_evo_meta(X, y, input_size=input_size, output_size=len(np.unique(y)), task=task)
    print("[✅ evo_meta_learner] 학습 완료 및 모델 저장됨")

def _load_meta():
    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    return meta

def predict_evo_meta(X_new, input_size):
    """
    predict.py에서 불러 쓰는 진입점.
    현재 저장된 메타가 task!='class'면 '클래스 선택' 용도가 아니므로 None 반환(안전 가드).
    """
    if not os.path.exists(MODEL_PATH):
        print("[❌ evo_meta_learner] 모델 없음")
        return None

    meta = _load_meta()
    task = meta.get("task", "class")  # 메타 없으면 class로 간주(과거 호환)
    if task != "class":
        # 전략 선택 모델을 클래스 선택에 잘못 쓰지 않도록 차단
        print(f"[ℹ️ evo_meta_learner] task={task} → class 예측에 사용하지 않음")
        return None

    out_size = int(meta.get("output_size", 3))
    model = EvoMetaModel(input_size, output_size=out_size)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_new, dtype=torch.float32)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        best = int(torch.argmax(probs, dim=1).item())
        return best

# ✅ 간단 휴리스틱: 실패 확률 기반 전략 추천(옵션)
def get_best_strategy_by_failure_probability(symbol, current_strategy, feature_tensor, model_outputs):
    """
    최근 심볼별 전략 성공률을 보고 현재 전략이 0.25 미만이고,
    다른 전략 중 0.45 이상이 있으면 그 전략을 제안. 근거 없으면 None.
    """
    try:
        path = "/persistent/prediction_log.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["symbol"] == symbol]
        df = df[df["status"].isin(["success","fail","v_success","v_fail"])]
        if df.empty:
            return None
        sr = df.pivot_table(index="strategy", values="status",
                            aggfunc=lambda s: (s.isin(["success","v_success"])).mean())
        sr = sr["status"].to_dict()
        curr = float(sr.get(current_strategy, 0.0))
        if curr < 0.25:
            alt = sorted([(k, float(v)) for k, v in sr.items() if k != current_strategy],
                         key=lambda x: x[1], reverse=True)
            if alt and alt[0][1] >= 0.45:
                return alt[0][0]
        return None
    except Exception:
        return None
