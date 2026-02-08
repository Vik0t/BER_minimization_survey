import torch.nn as nn
from src.config import CFG

class LSTM_EQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 2 * CFG.context_k + 1
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=CFG.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=CFG.dropout
        )
        self.fc = nn.Linear(CFG.hidden_dim, 16)

    def forward(self, x):
        # Reshape input to (batch, sequence, features)
        x = x.view(-1, self.seq_len, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, x.size(1)//2])
