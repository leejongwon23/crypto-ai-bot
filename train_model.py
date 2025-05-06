# train_model.py
import torch
import numpy as np
from model import LSTMModel

def train_model():
    model = LSTMModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ✅ 예시 데이터 (sin 파형을 정규화해 학습)
    x_train = np.sin(np.linspace(0, 100, 500))
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    
    inputs = torch.tensor(x_train[:-1].reshape(-1, 1, 1), dtype=torch.float32)
    targets = torch.tensor(x_train[1:].reshape(-1, 1), dtype=torch.float32)

    for epoch in range(10):  # 에폭 수는 향후 조절 가능
        output = model(inputs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ✅ 모델 저장
    torch.save(model.state_dict(), "best_model.pt")
    print("✅ 모델 학습 완료 및 저장됨")
