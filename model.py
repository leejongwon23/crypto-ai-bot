import torch
import torch.nn as nn

class GRUBiLSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, output_size=1):
        super(GRUBiLSTMModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.bilstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        lstm_out, _ = self.bilstm(gru_out)
        out = self.fc(lstm_out[:, -1, :])
        return out

def get_model(input_size=10):
    model = GRUBiLSTMModel(input_size=input_size)
    return model
