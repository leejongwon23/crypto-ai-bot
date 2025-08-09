import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "/persistent/models/evo_meta_learner.pt"

class EvoMetaModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def prepare_evo_meta_dataset(path="/persistent/wrong_predictions.csv", min_samples=50):
    if not os.path.exists(path):
        print(f"[❌ prepare_evo_meta_dataset] 파일 없음: {path}")
        return None, None
    df = pd.read_csv(path)
    if len(df) < min_samples:
        print(f"[❌ prepare_evo_meta_dataset] 샘플 부족: {len(df)}개")
        return None, None

    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            sm = eval(row.get("softmax") or "[]")
            if not sm or len(sm) != 3:
                continue
            expected_returns = eval(row.get("expected_returns") or "[0,0,0]")
            predicted_classes = eval(row.get("model_predictions") or "[0,0,0]")
            features = []
            for i in range(3):
                f = [sm[i], expected_returns[i], 1 if predicted_classes[i] == row.get("label", -1) else 0]
                features.extend(f)
            X_list.append(features)
            y_list.append(int(row.get("best_strategy", 0)))
        except Exception as e:
            continue
    if not X_list or not y_list:
        print("[❌ prepare_evo_meta_dataset] 유효 샘플 부족")
        return None, None
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[✅ prepare_evo_meta_dataset] X:{X.shape}, y:{y.shape}")
    return X, y

def train_evo_meta(X, y, input_size, num_strategies=3, epochs=10, batch_size=32, lr=1e-3):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvoMetaModel(input_size, output_size=num_strategies).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[evo_meta_learner] 학습 시작 → 샘플 수: {len(dataset)}, 전략 개수: {num_strategies}, 입력 크기: {input_size}")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        print(f"[evo_meta_learner] Epoch {epoch+1}/{epochs} → Loss: {epoch_loss:.4f}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    meta_info = {"input_size": input_size, "num_strategies": num_strategies}
    with open(MODEL_PATH.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    print(f"[✅ evo_meta_learner] 학습 완료 → {MODEL_PATH} (전략 개수={num_strategies})")
    return model

def train_evo_meta_loop(min_samples=50, auto_trigger=False):
    X, y = prepare_evo_meta_dataset(min_samples=min_samples)
    if X is None or y is None:
        if auto_trigger:
            print("[⏭️ evo_meta_learner] 실패 데이터 부족 → 자동 학습 스킵")
        return
    input_size = X.shape[1]
    print(f"[🚀 evo_meta_learner] 학습 시작 → 입력크기:{input_size}, 샘플:{len(X)}")
    train_evo_meta(X, y, input_size)
    print("[✅ evo_meta_learner] 학습 완료 및 모델 저장됨]")

def predict_evo_meta(X_new, input_size):
    if not os.path.exists(MODEL_PATH):
        print("[❌ evo_meta_learner] 모델 없음")
        return None
    model = EvoMetaModel(input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_new, dtype=torch.float32)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        best = torch.argmax(probs, dim=1).item()
        return best

# ✅ 누락: 실패 확률 기반 전략 추천 (간단 휴리스틱)
def get_best_strategy_by_failure_probability(symbol, current_strategy, feature_tensor, model_outputs):
    """
    간단 정책:
    - 최근 실패가 현재 전략에서 많고, 다른 전략(중기/장기 vs 단기 등)이 상대적으로 실패 적으면 교체 제안
    - 데이터가 부족하거나 근거 없으면 None 반환(=교체 안 함)
    """
    try:
        path = "/persistent/prediction_log.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df[df["symbol"] == symbol]
        df = df[df["status"].isin(["success","fail","v_success","v_fail"])]
        if df.empty: return None
        sr = df.pivot_table(index="strategy", values="status", aggfunc=lambda s: (s.isin(["success","v_success"])).mean())
        sr = sr["status"].to_dict()
        # 현재 전략 성공률이 0.25 미만이고, 다른 전략 중 0.45 이상이 있으면 그쪽으로
        curr = sr.get(current_strategy, 0.0)
        if curr < 0.25:
            alt = [(k,v) for k,v in sr.items() if k != current_strategy]
            if not alt: return None
            alt.sort(key=lambda x: x[1], reverse=True)
            if alt[0][1] >= 0.45:
                return alt[0][0]
        return None
    except Exception as e:
        return None
