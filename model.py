import torch
import torch.nn as nn

# ✅ LSTM 모델
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ✅ GRU 모델
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ✅ 공통 모델 불러오기 함수
def get_model(input_size=7, model_type="lstm"):
    if model_type == "gru":
        return GRUModel(input_size=input_size)
    else:
        return LSTMModel(input_size=input_size)
