import torch
import torch.nn as nn
import numpy as np
from model import GRUModel
import os

def train_gru_model(model_path="gru_model.pt"):
    # 예시용 데이터 생성 (실전에서는 실제 데이터로 교체)
    x = np.sin(np.linspace(0, 100, 500))
    x = (x - x.min()) / (x.max() - x.min())  # 정규화

    inputs = torch.tensor(x[:-1].reshape(-1, 1, 1), dtype=torch.float32)
    targets = torch.tensor(x[1:].reshape(-1, 1), dtype=torch.float32)

    model = GRUModel(input_size=1, hidden_size=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🧠 GRU 모델 학습 시작...")
    for epoch in range(20):
        output = model(inputs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"📚 epoch {epoch+1}, loss = {loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ GRU 모델 저장 완료: {model_path}")

if __name__ == "__main__":
    train_gru_model()
