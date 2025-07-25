# evo_meta_learner.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

MODEL_PATH = "/persistent/models/evo_meta_learner.pt"

# ✅ 진화형 메타러너 구조 (간단한 FeedForward)
class EvoMetaModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 3개 전략 softmax 선택
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ✅ 학습 함수
def train_evo_meta(X, y, input_size, epochs=10, batch_size=32, lr=1e-3):
    model = EvoMetaModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[✅ evo_meta_learner] 학습 완료 → {MODEL_PATH}")
    return model

# ✅ 예측 함수 (예측 실패 가능성 최소 전략 선택)
def predict_evo_meta(X_new, input_size):
    if not os.path.exists(MODEL_PATH):
        print("[❌ evo_meta_learner] 모델 없음")
        return None

    model = EvoMetaModel(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        x = torch.tensor(X_new, dtype=torch.float32)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        best = torch.argmax(probs, dim=1).item()
        return best  # 선택된 전략 인덱스

import pandas as pd
import numpy as np
import os

def prepare_evo_meta_dataset(path="/persistent/wrong_predictions.csv", min_samples=50):
    """
    실패 로그 기반으로 전략별 softmax, expected return 등을 학습용 X로 만들고,
    가장 적게 실패한 전략 인덱스를 y로 지정
    """
    if not os.path.exists(path):
        print(f"[❌ prepare_evo_meta_dataset] 파일 없음: {path}")
        return None, None

    df = pd.read_csv(path)
    if len(df) < min_samples:
        print(f"[❌ prepare_evo_meta_dataset] 샘플 부족: {len(df)}개")
        return None, None

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        try:
            # ✅ softmax 값 파싱
            sm = eval(row.get("softmax") or "[]")
            if not sm or len(sm) != 3:
                continue

            # ✅ 전략별 softmax, expected return, 예측 클래스, 실제 수익률 등 구성
            expected_returns = eval(row.get("expected_returns") or "[0,0,0]")
            predicted_classes = eval(row.get("model_predictions") or "[0,0,0]")
            actual_return = float(row.get("return") or 0)

            # ✅ 각 전략별 특성 벡터 구성
            features = []
            for i in range(3):
                f = [
                    sm[i],                     # softmax
                    expected_returns[i],       # 기대 수익률
                    1 if predicted_classes[i] == row["label"] else 0,  # 예측 적중 여부
                ]
                features.extend(f)

            X_list.append(features)

            # ✅ 실패율이 가장 낮은 전략을 정답으로 설정
            best_strategy = int(row.get("best_strategy", 0))
            y_list.append(best_strategy)

        except Exception as e:
            print(f"[⚠️ prepare_evo_meta_dataset] 예외 발생: {e}")
            continue

    if not X_list or not y_list:
        print("[❌ prepare_evo_meta_dataset] 유효 샘플 부족")
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[✅ prepare_evo_meta_dataset] X:{X.shape}, y:{y.shape}")
    return X, y

from evo_meta_learner import train_evo_meta, EvoMetaModel
from evo_meta_dataset import prepare_evo_meta_dataset  # 경로에 따라 조정
import os

def train_evo_meta_loop(min_samples=50, input_size=9):
    X, y = prepare_evo_meta_dataset(min_samples=min_samples)
    if X is None or y is None:
        print("[ℹ️ train_evo_meta_loop] 학습 데이터 부족으로 학습 건너뜀")
        return
    train_evo_meta(X, y, input_size=input_size)
