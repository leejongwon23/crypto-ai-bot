import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 50)
        c0 = torch.zeros(1, x.size(0), 50)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[-1])
        return out

# 🔁 저장/불러오기 함수 추가
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path="model.pth"):
    model = LSTMModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
