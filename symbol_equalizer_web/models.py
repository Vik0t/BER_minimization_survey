# models.py
import torch
import torch.nn as nn
from typing import Dict, Any

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(lstm_output)
        attention_weights = self.softmax(attention_weights)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context

class LSTMRxEqualizer(nn.Module):
    def __init__(self, input_dim: int = 2, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Параметры из конфига
        self.context_k = self.config.get('CONTEXT_K', 40)
        self.hidden_dim = self.config.get('HIDDEN_DIM', 256)
        self.dropout = self.config.get('DROPOUT', 0.3)
        self.lstm_layers = self.config.get('LSTM_LAYERS', 2)
        self.lstm_hidden = self.config.get('LSTM_HIDDEN', 128)
        self.bidirectional = self.config.get('BIDIRECTIONAL', True)
        self.use_attention = self.config.get('USE_ATTENTION', True)
        
        self.seq_len = 2 * self.context_k + 1
        self.input_dim = input_dim

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5)
        )

        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0
        )

        # Механизм внимания
        lstm_output_dim = self.lstm_hidden * (2 if self.bidirectional else 1)
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_dim)

        # BatchNorm после LSTM
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        # Классификатор
        classifier_input_dim = lstm_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(self.hidden_dim // 2, 16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.input_dim)
        
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            context = self.attention(lstm_out)
        else:
            if self.bidirectional:
                forward_hidden = hidden[-2, :, :]
                backward_hidden = hidden[-1, :, :]
                context = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                context = hidden[-1, :, :]
        
        context = self.lstm_norm(context)
        output = self.classifier(context)
        
        return output

class HybridCNN_LSTM_Equalizer(nn.Module):
    def __init__(self, input_dim: int = 2, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        self.context_k = self.config.get('CONTEXT_K', 40)
        self.hidden_dim = self.config.get('HIDDEN_DIM', 256)
        self.dropout = self.config.get('DROPOUT', 0.3)
        self.lstm_hidden = self.config.get('LSTM_HIDDEN', 128)
        self.bidirectional = self.config.get('BIDIRECTIONAL', True)
        
        self.seq_len = 2 * self.context_k + 1

        # CNN часть
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.3),
        )

        # LSTM часть
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )

        lstm_output_dim = self.lstm_hidden * (2 if self.bidirectional else 1)

        # Attention
        self.attention = AttentionLayer(lstm_output_dim)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Reshape для CNN
        x = x.view(batch_size, self.seq_len, 2)
        x = x.transpose(1, 2)  # (batch, 2, seq_len)

        # CNN
        cnn_features = self.cnn(x)  # (batch, 256, seq_len)
        cnn_features = cnn_features.transpose(1, 2)  # (batch, seq_len, 256)

        # LSTM
        lstm_out, _ = self.lstm(cnn_features)

        # Attention
        context = self.attention(lstm_out)

        # Классификация
        output = self.classifier(context)

        return output