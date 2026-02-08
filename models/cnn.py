import torch
import torch.nn as nn
from src.config import CFG

class EnhancedCNNRxEqualizer(nn.Module):
    """
    Усиленная CNN архитектура с:
    - Большим контекстом (k=4 → 9 символов)
    - Параллельными свёртками для разных масштабов зависимостей
    - Spatial dropout для регуляризации временных зависимостей
    """
    def __init__(self):
        super().__init__()
        self.seq_len = 2 * CFG.context_k + 1
        self.channels = 2
        k = CFG.context_k

        # Параллельные ветки с разными размерами ядер
        self.branch1 = nn.Sequential(
            nn.Conv1d(self.channels, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(CFG.dropout)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(self.channels, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(CFG.dropout)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(self.channels, 96, kernel_size=7, padding=3),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(CFG.dropout)
        )

        # Комбинирование веток
        self.combine = nn.Sequential(
            nn.Conv1d(288, 256, kernel_size=1),  # 96*3 = 288
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(CFG.dropout)
        )

        # Глубокая обработка
        self.deep_conv = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(CFG.dropout),

            nn.Conv1d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(CFG.dropout)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, CFG.hidden_dim),
            nn.LayerNorm(CFG.hidden_dim),
            nn.GELU(),
            nn.Dropout(CFG.dropout * 1.5),  # Более сильная регуляризация на выходе
            nn.Linear(CFG.hidden_dim, 16)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.channels, self.seq_len)

        # Параллельные ветки
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # Конкатенация и обработка
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.combine(x)
        x = self.deep_conv(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
