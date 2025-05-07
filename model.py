import torch
import torch.nn as nn

# ✅ LSTM 모델 클래스 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 타임스텝 출력
        out = self.fc(out)
        return self.sigmoid(out)

# ✅ 외부에서 사용할 수 있도록 모델 생성 함수 제공
def get_model(input_size):
    return LSTMModel(input_size)
