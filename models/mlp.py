import torch.nn as nn
from src.config import CFG

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        dim = (2 * CFG.context_k + 1) * 2
        self.net = nn.Sequential(
            nn.Linear(dim, CFG.hidden_dim),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_dim, CFG.hidden_dim),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_dim, 16)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
