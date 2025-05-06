# train_model.py
import torch
from model import LSTMModel
import numpy as np

def train_model():
    model = LSTMModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 예시 학습 데이터 (테스트용)
    x_train = np.sin(np.linspace(0, 100, 500))
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    inputs = torch.tensor(x_train[:-1].reshape(-1, 1, 1), dtype=torch.float32)
    targets = torch.tensor(x_train[1:].reshape(-1, 1), dtype=torch.float32)

    for epoch in range(10):  # 에폭 수는 테스트용
        output = model(inputs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "best_model.pt")
    print("✅ 모델 저장 완료")
