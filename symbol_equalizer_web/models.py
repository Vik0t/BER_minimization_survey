import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        attention_weights = self.softmax(self.attention(lstm_output))
        return torch.sum(attention_weights * lstm_output, dim=1)


class LSTMRxEqualizer(nn.Module):
    def __init__(self, input_dim: int = 2, config: Dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

        self.context_k = self.config.get("CONTEXT_K", 40)
        self.hidden_dim = self.config.get("HIDDEN_DIM", 256)
        self.dropout = self.config.get("DROPOUT", 0.3)
        self.lstm_layers = self.config.get("LSTM_LAYERS", 2)
        self.lstm_hidden = self.config.get("LSTM_HIDDEN", 128)
        self.bidirectional = self.config.get("BIDIRECTIONAL", True)
        self.use_attention = self.config.get("USE_ATTENTION", True)

        self.seq_len = 2 * self.context_k + 1
        self.input_dim = input_dim

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0,
        )

        lstm_output_dim = self.lstm_hidden * (2 if self.bidirectional else 1)
        self.attention = AttentionLayer(lstm_output_dim)
        self.center_fusion = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, lstm_output_dim),
            nn.LayerNorm(lstm_output_dim),
            nn.GELU(),
        )
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(self.hidden_dim // 2, self.config.get("OUTPUT_DIM", 2)),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
                gate = self.lstm_hidden
                param.data[gate : 2 * gate] = 1.0
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.seq_len, self.input_dim)
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        center_feature = lstm_out[:, self.context_k, :]

        if self.use_attention:
            context = self.attention(lstm_out)
        elif self.bidirectional:
            context = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            context = hidden[-1]

        context = self.center_fusion(torch.cat([context, center_feature], dim=1))
        context = self.lstm_norm(context)
        return self.classifier(context)


class HybridCNNLSTMEqualizer(nn.Module):
    def __init__(self, input_dim: int = 2, config: Dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

        self.context_k = self.config.get("CONTEXT_K", 40)
        self.hidden_dim = self.config.get("HIDDEN_DIM", 256)
        self.dropout = self.config.get("DROPOUT", 0.3)
        self.lstm_hidden = self.config.get("LSTM_HIDDEN", 128)
        self.bidirectional = self.config.get("BIDIRECTIONAL", True)
        self.seq_len = 2 * self.context_k + 1
        self.input_dim = input_dim

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

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )

        lstm_output_dim = self.lstm_hidden * (2 if self.bidirectional else 1)
        self.attention = AttentionLayer(lstm_output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.config.get("OUTPUT_DIM", 16)),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.seq_len, self.input_dim).transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float, conv_kernel: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_dim * 2)
        self.ff_out = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout(attn_out)

        local_in = self.norm2(x).transpose(1, 2)
        local_out = self.local_mixer(local_in).transpose(1, 2)
        x = x + self.dropout(local_out)

        ff_in = self.norm3(x)
        value, gate = self.ff_in(ff_in).chunk(2, dim=-1)
        ff_out = self.ff_out(value * F.gelu(gate))
        return x + self.dropout(ff_out)


class TransformerRxEqualizer(nn.Module):
    def __init__(self, input_dim: int = 2, config: Dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

        self.context_k = self.config.get("CONTEXT_K", 20)
        self.seq_len = 2 * self.context_k + 1
        self.input_dim = input_dim
        self.dim = self.config.get("TRANSFORMER_DIM", 128)
        self.dropout = self.config.get("DROPOUT", 0.2)
        self.hidden_dim = self.config.get("HIDDEN_DIM", 64)
        self.ff_dim = self.config.get("TRANSFORMER_FF_DIM", 256)
        self.heads = self.config.get("TRANSFORMER_HEADS", 4)
        self.layers = self.config.get("TRANSFORMER_LAYERS", 4)
        self.conv_kernel = self.config.get("TRANSFORMER_CONV_KERNEL", 3)

        self.input_proj = nn.Linear(self.input_dim, self.dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))
        self.input_dropout = nn.Dropout(self.dropout * 0.5)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=self.dim,
                    heads=self.heads,
                    ff_dim=self.ff_dim,
                    dropout=self.dropout,
                    conv_kernel=self.conv_kernel,
                )
                for _ in range(self.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.dim)
        self.center_fusion = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(self.hidden_dim // 2, self.config.get("OUTPUT_DIM", 2)),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.seq_len, self.input_dim)
        x = self.input_proj(x)
        x = self.input_dropout(x + self.pos_embedding)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        center = x[:, self.context_k, :]
        global_context = x.mean(dim=1)
        fused = self.center_fusion(torch.cat([center, global_context], dim=1))
        return self.regressor(fused)


class MLPRxEqualizer(nn.Module):
    def __init__(self, input_dim: int = 2, config: Dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}
        self.context_k = self.config.get("CONTEXT_K", 20)
        self.seq_len = 2 * self.context_k + 1
        self.input_dim = input_dim
        self.hidden_dim = self.config.get("HIDDEN_DIM", 64)
        self.dropout = self.config.get("DROPOUT", 0.2)

        flat_input_dim = self.seq_len * self.input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(self.hidden_dim // 2, self.config.get("OUTPUT_DIM", 2)),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


HybridCNN_LSTM_Equalizer = HybridCNNLSTMEqualizer


MODEL_REGISTRY = {
    "lstm": LSTMRxEqualizer,
    "hybrid": HybridCNNLSTMEqualizer,
    "cnn_lstm": HybridCNNLSTMEqualizer,
    "transformer": TransformerRxEqualizer,
    "mlp": MLPRxEqualizer,
}


def build_model(model_type: str, input_dim: int = 2, config: Dict[str, Any] | None = None) -> nn.Module:
    normalized_type = model_type.lower()
    if normalized_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[normalized_type](input_dim=input_dim, config=config)
