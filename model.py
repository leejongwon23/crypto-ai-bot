import torch
import torch.nn as nn

class CryptoPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(CryptoPredictor, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        gru_out, _ = self.gru(lstm_out)
        out = self.fc(gru_out[:, -1, :])
        return self.sigmoid(out)
