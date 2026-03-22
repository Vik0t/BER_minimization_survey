import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Config:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    DATA_DIR_CANDIDATES = [Path("symbols_new"), Path("Symbols_1m_1ch_PR"), Path(".")]
    MAX_FILES = 32
    TRAIN_PORTION = 0.97

    CONTEXT_K = 20
    SEQ_LEN = 2 * CONTEXT_K + 1
    INPUT_DIM = 2

    HIDDEN_DIM = 64
    DROPOUT = 0.2
    LSTM_HIDDEN = 64
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    TRANSFORMER_DIM = 128
    TRANSFORMER_LAYERS = 4
    TRANSFORMER_HEADS = 4
    TRANSFORMER_FF_DIM = 256
    TRANSFORMER_CONV_KERNEL = 3
    COMPLEX_CHANNELS = 24
    COMPLEX_BLOCK_CHANNELS = [24, 32]
    COMPLEX_KERNEL_SIZES = [5, 7]
    COMPLEX_HEAD_DIM = 128
    COMPLEX_TEMPORAL_DIM = 96
    COMPLEX_TEMPORAL_DILATIONS = [1, 2, 4]
    COMPLEX_SEQ_DIM = 96
    COMPLEX_LSTM_HIDDEN = 64
    COMPLEX_LSTM_LAYERS = 2
    COMPLEX_USE_KERR = False
    COMPLEX_KERR_KERNEL = 5
    COMPLEX_KERR_INIT_GAMMA = 0.02
    DBP_NUM_STEPS = 8
    DBP_KERNEL_SIZE = 7
    DBP_USE_SYMMETRIC_FILTER = True
    DBP_SEQSTAT_DIM = 128

    EPOCHS = 250
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    TRAIN_BLOCK_SIZE = 32768
    EVAL_BATCH_SIZE = 131072
    MIN_BLOCK_SIZE = 4096
    USE_AMP = True
    USE_TORCH_COMPILE = False
    TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
    LR_SCHEDULER = "plateau"
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 8
    SCHEDULER_THRESHOLD = 1e-4

    DECAY_STEPS = 20
    MIN_LR = 1e-5
    LOG_EVERY = 1
    TEST_BER_EVERY = 5

    OUT_DIR = Path("clean_compare_outputs")
    MODEL_TYPES = ["complex_lstm", "complex_dbp_seqstat", "complex_cnn_lstm", "complex_cnn"]
    SAVE_BEST = True
    RUN_SWEEP_EXPERIMENTS = False
    SWEEP_TEST_FILES = 1
    WINDOW_SWEEP_VALUES = [2, 4, 8, 12, 16, 20]
    HIDDEN_SWEEP_VALUES = [8, 16, 32, 64]


if Config.DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


CONSTELLATION = torch.tensor(
    [
        [-0.948683, -0.948683],
        [-0.948683, -0.316228],
        [-0.948683, 0.316228],
        [-0.948683, 0.948683],
        [-0.316228, -0.948683],
        [-0.316228, -0.316228],
        [-0.316228, 0.316228],
        [-0.316228, 0.948683],
        [0.316228, -0.948683],
        [0.316228, -0.316228],
        [0.316228, 0.316228],
        [0.316228, 0.948683],
        [0.948683, -0.948683],
        [0.948683, -0.316228],
        [0.948683, 0.316228],
        [0.948683, 0.948683],
    ],
    dtype=torch.float32,
)

BIT_LABELS = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 0],
    ],
    dtype=torch.uint8,
)


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
        weights = self.softmax(self.attention(lstm_output))
        return torch.sum(weights * lstm_output, dim=1)


class LSTMRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        lstm_out_dim = Config.LSTM_HIDDEN * (2 if Config.BIDIRECTIONAL else 1)
        self.embedding = nn.Sequential(
            nn.Linear(Config.INPUT_DIM, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=Config.LSTM_HIDDEN,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.LSTM_LAYERS > 1 else 0.0,
        )
        self.use_attention = Config.USE_ATTENTION
        self.attention = AttentionLayer(lstm_out_dim)
        self.center_fusion = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, lstm_out_dim),
            nn.LayerNorm(lstm_out_dim),
            nn.GELU(),
        )
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM // 2, 2),
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
                gate = Config.LSTM_HIDDEN
                param.data[gate : 2 * gate] = 1.0
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        center_feature = lstm_out[:, Config.CONTEXT_K, :]
        if self.use_attention:
            context = self.attention(lstm_out)
        elif Config.BIDIRECTIONAL:
            context = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            context = hidden[-1]
        context = self.center_fusion(torch.cat([context, center_feature], dim=1))
        context = self.lstm_norm(context)
        return self.classifier(context)


class HybridCNNLSTMEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.3),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=Config.LSTM_HIDDEN,
            num_layers=2,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT,
        )
        out_dim = Config.LSTM_HIDDEN * (2 if Config.BIDIRECTIONAL else 1)
        self.attention = AttentionLayer(out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM).transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x)


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1, symmetric: bool = False):
        super().__init__()
        self.symmetric = symmetric
        effective_kernel = 2 * kernel_size - 1 if symmetric else kernel_size
        padding = effective_kernel // 2
        self.real_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.imag_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.real_conv.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.imag_conv.weight, nonlinearity="linear")

    def _build_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.symmetric:
            return weight
        return torch.cat([weight, weight.flip(dims=(2,))[:, :, 1:]], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0::2, :]
        imag = x[:, 1::2, :]
        real_weight = self._build_weight(self.real_conv.weight)
        imag_weight = self._build_weight(self.imag_conv.weight)
        out_real = F.conv1d(real, real_weight, padding=self.real_conv.padding[0], groups=self.real_conv.groups) - F.conv1d(
            imag, imag_weight, padding=self.imag_conv.padding[0], groups=self.imag_conv.groups
        )
        out_imag = F.conv1d(real, imag_weight, padding=self.imag_conv.padding[0], groups=self.imag_conv.groups) + F.conv1d(
            imag, real_weight, padding=self.real_conv.padding[0], groups=self.real_conv.groups
        )
        return torch.stack((out_real, out_imag), dim=2).flatten(1, 2)


class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_norm = nn.BatchNorm1d(2 * in_channels)
        self.expand = ComplexConv1d(in_channels, out_channels, kernel_size=1)
        self.expand_norm = nn.BatchNorm1d(2 * out_channels)
        self.depthwise = ComplexConv1d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels)
        self.depthwise_norm = nn.BatchNorm1d(2 * out_channels)
        self.kerr = KerrLikeActivation(out_channels) if Config.COMPLEX_USE_KERR else nn.Identity()
        self.project = ComplexConv1d(out_channels, out_channels, kernel_size=1)
        self.project_norm = nn.BatchNorm1d(2 * out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(Config.DROPOUT * 0.5)
        self.skip = ComplexConv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.pre_norm(x)
        x = self.activation(self.expand_norm(self.expand(x)))
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        x = self.kerr(x)
        x = self.dropout(self.project_norm(self.project(x)))
        return residual + x


class KerrLikeActivation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernel_size = Config.COMPLEX_KERR_KERNEL
        self.power_filter = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1), Config.COMPLEX_KERR_INIT_GAMMA))
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.power_filter.weight)
        center = self.power_filter.weight.size(-1) // 2
        self.power_filter.weight.data[:, :, center] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0::2, :]
        imag = x[:, 1::2, :]
        power = self.power_filter(real.square() + imag.square())
        phase = self.gamma * power
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        out_real = cos_phase * real + sin_phase * imag
        out_imag = cos_phase * imag - sin_phase * real
        return torch.stack((out_real, out_imag), dim=2).flatten(1, 2)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.BatchNorm1d(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(Config.DROPOUT * 0.5)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm(x))
        x = self.depthwise(x)
        x = self.activation(self.pointwise(x))
        x = self.dropout(x)
        return x + residual


class ComplexFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = Config.COMPLEX_BLOCK_CHANNELS
        kernels = Config.COMPLEX_KERNEL_SIZES
        if len(channels) != len(kernels):
            raise ValueError("COMPLEX_BLOCK_CHANNELS and COMPLEX_KERNEL_SIZES must have the same length.")

        self.stem = ComplexConv1d(1, channels[0], kernel_size=1)
        self.stem_norm = nn.BatchNorm1d(2 * channels[0])
        self.blocks = nn.ModuleList()
        in_channels = channels[0]
        for out_channels, kernel_size in zip(channels, kernels):
            self.blocks.append(ComplexResidualBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels

        self.final_norm = nn.BatchNorm1d(2 * in_channels)
        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        x = raw.transpose(1, 2)
        x = self.stem_norm(self.stem(x))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        real = x[:, 0::2, :]
        imag = x[:, 1::2, :]
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-6)
        seq_features = torch.cat([real, imag, magnitude], dim=1).transpose(1, 2)
        return raw, x, seq_features


class ComplexDBPStep1Ch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = ComplexConv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=Config.DBP_KERNEL_SIZE,
            symmetric=Config.DBP_USE_SYMMETRIC_FILTER,
        )
        self.nonlinear = KerrLikeActivation(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class ComplexDBPSeqStatRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = nn.ModuleList([ComplexDBPStep1Ch() for _ in range(Config.DBP_NUM_STEPS)])
        feature_dim = 3 * (Config.DBP_NUM_STEPS + 1)
        fused_dim = feature_dim * 3 + Config.INPUT_DIM
        self.head = nn.Sequential(
            nn.Linear(fused_dim, Config.DBP_SEQSTAT_DIM),
            nn.LayerNorm(Config.DBP_SEQSTAT_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.DBP_SEQSTAT_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _seq_features(x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-6)
        return torch.cat([real, imag, magnitude], dim=1).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        state = raw.transpose(1, 2)
        features = [self._seq_features(state)]
        for step in self.steps:
            state = step(state)
            features.append(self._seq_features(state))
        seq = torch.cat(features, dim=2)
        center = seq[:, Config.CONTEXT_K, :]
        mean = seq.mean(dim=1)
        std = seq.std(dim=1, unbiased=False)
        fused = torch.cat([center, mean, std, raw[:, Config.CONTEXT_K, :]], dim=1)
        return self.head(fused)


class ComplexCNNRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ComplexFeatureEncoder()
        temporal_in_dim = 3 * self.encoder.out_channels
        self.temporal_proj = nn.Conv1d(temporal_in_dim, Config.COMPLEX_TEMPORAL_DIM, kernel_size=1, bias=False)
        self.temporal_blocks = nn.ModuleList(
            [TemporalConvBlock(Config.COMPLEX_TEMPORAL_DIM, dilation) for dilation in Config.COMPLEX_TEMPORAL_DILATIONS]
        )
        self.temporal_norm = nn.BatchNorm1d(Config.COMPLEX_TEMPORAL_DIM)
        self.pool_score = nn.Conv1d(Config.COMPLEX_TEMPORAL_DIM, 1, kernel_size=1)
        fused_dim = Config.COMPLEX_TEMPORAL_DIM * 2 + 2 * Config.INPUT_DIM
        self.head = nn.Sequential(
            nn.Linear(fused_dim, Config.COMPLEX_HEAD_DIM),
            nn.LayerNorm(Config.COMPLEX_HEAD_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.COMPLEX_HEAD_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, _, seq_features = self.encoder(x)
        raw_center = raw[:, Config.CONTEXT_K, :]
        raw_global = raw.mean(dim=1)
        temporal = seq_features.transpose(1, 2)
        temporal = self.temporal_proj(temporal)
        for block in self.temporal_blocks:
            temporal = block(temporal)
        temporal = self.temporal_norm(temporal)

        center = temporal[:, :, Config.CONTEXT_K]
        scores = self.pool_score(temporal)
        weights = torch.softmax(scores, dim=2)
        global_context = torch.sum(weights * temporal, dim=2)
        fused = torch.cat([center, global_context, raw_center, raw_global], dim=1)
        return self.head(fused)


class ComplexLSTMRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ComplexFeatureEncoder()
        seq_in_dim = 3 * self.encoder.out_channels
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_in_dim, Config.COMPLEX_SEQ_DIM),
            nn.LayerNorm(Config.COMPLEX_SEQ_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=Config.COMPLEX_SEQ_DIM,
            hidden_size=Config.COMPLEX_LSTM_HIDDEN,
            num_layers=Config.COMPLEX_LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.COMPLEX_LSTM_LAYERS > 1 else 0.0,
        )
        lstm_out_dim = Config.COMPLEX_LSTM_HIDDEN * (2 if Config.BIDIRECTIONAL else 1)
        self.attention = AttentionLayer(lstm_out_dim)
        self.center_fusion = nn.Sequential(
            nn.Linear(lstm_out_dim * 2 + Config.INPUT_DIM, Config.COMPLEX_HEAD_DIM),
            nn.LayerNorm(Config.COMPLEX_HEAD_DIM),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(Config.COMPLEX_HEAD_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM // 2, 2),
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
                gate = Config.COMPLEX_LSTM_HIDDEN
                param.data[gate : 2 * gate] = 1.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, _, seq_features = self.encoder(x)
        seq = self.seq_proj(seq_features)
        lstm_out, _ = self.lstm(seq)
        center = lstm_out[:, Config.CONTEXT_K, :]
        context = self.attention(lstm_out)
        fused = self.center_fusion(torch.cat([center, context, raw[:, Config.CONTEXT_K, :]], dim=1))
        return self.head(fused)


class ComplexCNNLSTMRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ComplexFeatureEncoder()
        seq_in_dim = 3 * self.encoder.out_channels
        self.temporal_proj = nn.Conv1d(seq_in_dim, Config.COMPLEX_TEMPORAL_DIM, kernel_size=1, bias=False)
        self.temporal_blocks = nn.ModuleList(
            [TemporalConvBlock(Config.COMPLEX_TEMPORAL_DIM, dilation) for dilation in Config.COMPLEX_TEMPORAL_DILATIONS[:2]]
        )
        self.temporal_norm = nn.BatchNorm1d(Config.COMPLEX_TEMPORAL_DIM)
        self.lstm = nn.LSTM(
            input_size=Config.COMPLEX_TEMPORAL_DIM,
            hidden_size=Config.COMPLEX_LSTM_HIDDEN,
            num_layers=Config.COMPLEX_LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.COMPLEX_LSTM_LAYERS > 1 else 0.0,
        )
        lstm_out_dim = Config.COMPLEX_LSTM_HIDDEN * (2 if Config.BIDIRECTIONAL else 1)
        self.attention = AttentionLayer(lstm_out_dim)
        self.center_fusion = nn.Sequential(
            nn.Linear(lstm_out_dim * 2 + Config.INPUT_DIM, Config.COMPLEX_HEAD_DIM),
            nn.LayerNorm(Config.COMPLEX_HEAD_DIM),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(Config.COMPLEX_HEAD_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM // 2, 2),
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
                gate = Config.COMPLEX_LSTM_HIDDEN
                param.data[gate : 2 * gate] = 1.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, _, seq_features = self.encoder(x)
        temporal = self.temporal_proj(seq_features.transpose(1, 2))
        for block in self.temporal_blocks:
            temporal = block(temporal)
        temporal = self.temporal_norm(temporal).transpose(1, 2)
        lstm_out, _ = self.lstm(temporal)
        center = lstm_out[:, Config.CONTEXT_K, :]
        context = self.attention(lstm_out)
        fused = self.center_fusion(torch.cat([center, context, raw[:, Config.CONTEXT_K, :]], dim=1))
        return self.head(fused)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=Config.DROPOUT,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=Config.TRANSFORMER_CONV_KERNEL,
                padding=Config.TRANSFORMER_CONV_KERNEL // 2,
                groups=dim,
            ),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_dim * 2)
        self.ff_out = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        with sdpa_context():
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
    def __init__(self):
        super().__init__()
        dim = Config.TRANSFORMER_DIM
        self.input_proj = nn.Linear(Config.INPUT_DIM, dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, dim))
        self.input_dropout = nn.Dropout(Config.DROPOUT * 0.5)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=dim,
                    heads=Config.TRANSFORMER_HEADS,
                    ff_dim=Config.TRANSFORMER_FF_DIM,
                )
                for _ in range(Config.TRANSFORMER_LAYERS)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.center_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM // 2, 2),
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
        x = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        x = self.input_proj(x)
        x = self.input_dropout(x + self.pos_embedding)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        center = x[:, Config.CONTEXT_K, :]
        global_context = x.mean(dim=1)
        fused = self.center_fusion(torch.cat([center, global_context], dim=1))
        return self.regressor(fused)


class MLPRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = Config.SEQ_LEN * Config.INPUT_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def log(msg: str):
    print(msg, flush=True)


def is_cuda_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def mark_cudagraph_step_begin():
    if Config.DEVICE.type != "cuda":
        return
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        compiler_mod.cudagraph_mark_step_begin()


def autocast_context():
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP,
    )


def sdpa_context():
    if Config.DEVICE.type != "cuda":
        return nullcontext()
    attention_mod = getattr(torch.nn, "attention", None)
    if attention_mod is not None and hasattr(attention_mod, "sdpa_kernel") and hasattr(attention_mod, "SDPBackend"):
        return attention_mod.sdpa_kernel([attention_mod.SDPBackend.MATH])
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is not None and hasattr(cuda_backends, "sdp_kernel"):
        return cuda_backends.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    return nullcontext()


def symbols_to_classes(symbols: torch.Tensor) -> torch.Tensor:
    constellation = CONSTELLATION.to(symbols.device)
    diff = symbols.unsqueeze(1) - constellation.unsqueeze(0)
    dist = torch.sum(diff.square(), dim=2)
    return torch.argmin(dist, dim=1)


def calculate_ber_from_classes(tx_classes: torch.Tensor, rx_classes: torch.Tensor) -> float:
    bit_labels = BIT_LABELS.to(tx_classes.device)
    tx_bits = bit_labels[tx_classes]
    rx_bits = bit_labels[rx_classes]
    return (tx_bits != rx_bits).float().mean().item()


def discover_symbol_files() -> Tuple[Path, List[int]]:
    for base_dir in Config.DATA_DIR_CANDIDATES:
        if not base_dir.exists():
            continue
        files = sorted(base_dir.glob("Symbols_1m_1ch_PR_*.csv"))
        if files:
            indices = sorted(int(path.stem.split("_")[-1]) for path in files)
            return base_dir, indices[: Config.MAX_FILES]
    raise FileNotFoundError("No Symbols_1m_1ch_PR_*.csv files found.")


def resolve_splits(file_indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
    train_files = max(1, int(len(file_indices) * Config.TRAIN_PORTION))
    if train_files >= len(file_indices):
        train_files = len(file_indices) - 1
    train_pool = file_indices[:train_files]
    if len(train_pool) < 2:
        raise ValueError("Need at least 2 train files so one can be held out for validation.")
    train_idx = train_pool[:-1]
    val_idx = [train_pool[-1]]
    test_idx = file_indices[train_files:]
    if not test_idx:
        raise ValueError("MXNet-style split produced no test files. Increase MAX_FILES.")
    return train_idx, val_idx, test_idx


def load_files(base_dir: Path, file_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tx_list: List[torch.Tensor] = []
    rx_list: List[torch.Tensor] = []
    for idx in file_indices:
        path = base_dir / f"Symbols_1m_1ch_PR_{idx}.csv"
        arr = pd.read_csv(path, header=None, dtype=np.float32).to_numpy(copy=False)
        tx_list.append(torch.from_numpy(arr[:, 0:2].copy()))
        rx_list.append(torch.from_numpy(arr[:, 2:4].copy()))
    return torch.cat(tx_list, dim=0), torch.cat(rx_list, dim=0)


def make_windows(rx_symbols: torch.Tensor, tx_symbols: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_rx = (rx_symbols - mean) / std
    norm_rx = torch.nan_to_num(norm_rx, nan=0.0, posinf=0.0, neginf=0.0)
    window_view = norm_rx.unfold(0, Config.SEQ_LEN, 1).permute(0, 2, 1).contiguous()
    x = window_view.view(window_view.size(0), -1).contiguous()
    y = tx_symbols[Config.CONTEXT_K : tx_symbols.size(0) - Config.CONTEXT_K].contiguous()
    return x, y


def prepare_data(max_test_files: Optional[int] = None) -> Dict[str, torch.Tensor]:
    base_dir, all_indices = discover_symbol_files()
    train_idx, val_idx, test_idx = resolve_splits(all_indices)
    if max_test_files is not None:
        test_idx = test_idx[:max_test_files]
        if not test_idx:
            raise ValueError("max_test_files truncated test split to zero files.")
    log(f"Data dir: {base_dir}")
    log(f"Train files: {train_idx}")
    log(f"Val files: {val_idx}")
    log(f"Test files: {test_idx}")

    tx_train, rx_train = load_files(base_dir, train_idx)
    tx_val, rx_val = load_files(base_dir, val_idx)
    tx_test, rx_test = load_files(base_dir, test_idx)

    mean = rx_train.mean(dim=0, keepdim=True)
    std = rx_train.std(dim=0, keepdim=True)
    std[std == 0] = 1.0

    train_x, train_y = make_windows(rx_train, tx_train, mean, std)
    val_x, val_y = make_windows(rx_val, tx_val, mean, std)
    test_x, test_y = make_windows(rx_test, tx_test, mean, std)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "tx_test": tx_test,
        "rx_test": rx_test,
        "mean": mean,
        "std": std,
    }


def make_model(name: str) -> nn.Module:
    if name == "lstm":
        return LSTMRxEqualizer().to(Config.DEVICE)
    if name in {"hybrid", "cnn_lstm"}:
        return HybridCNNLSTMEqualizer().to(Config.DEVICE)
    if name == "complex_lstm":
        return ComplexLSTMRxEqualizer().to(Config.DEVICE)
    if name == "complex_dbp_seqstat":
        return ComplexDBPSeqStatRxEqualizer().to(Config.DEVICE)
    if name == "complex_cnn_lstm":
        return ComplexCNNLSTMRxEqualizer().to(Config.DEVICE)
    if name == "complex_cnn":
        return ComplexCNNRxEqualizer().to(Config.DEVICE)
    if name == "transformer":
        return TransformerRxEqualizer().to(Config.DEVICE)
    if name == "mlp":
        return MLPRxEqualizer().to(Config.DEVICE)
    raise ValueError(f"Unknown model: {name}")


def iter_tensor_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
    total = x.size(0)
    if shuffle:
        order = torch.randperm(total)
        x = x.index_select(0, order)
        y = y.index_select(0, order)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        xb = x[start:end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
        yb = y[start:end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
        yield xb, yb


@torch.inference_mode()
def evaluate_split(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[float, float, float, int]:
    model.eval()
    criterion = nn.MSELoss()
    bit_labels = BIT_LABELS.to(Config.DEVICE)
    current_batch_size = batch_size

    while True:
        total_loss = 0.0
        total_samples = 0
        correct = 0
        bit_errors = 0
        total_bits = 0
        try:
            for xb, yb in iter_tensor_batches(x, y, batch_size=current_batch_size, shuffle=False):
                with autocast_context():
                    mark_cudagraph_step_begin()
                    preds = model(xb)
                    loss = criterion(preds, yb)
                pred_classes = symbols_to_classes(preds.float())
                target_classes = symbols_to_classes(yb.float())
                batch_size_now = yb.size(0)
                total_loss += loss.item() * batch_size_now
                total_samples += batch_size_now
                correct += (pred_classes == target_classes).sum().item()
                tx_bits = bit_labels[target_classes]
                rx_bits = bit_labels[pred_classes]
                bit_errors += (tx_bits != rx_bits).sum().item()
                total_bits += tx_bits.numel()
            return (
                total_loss / max(total_samples, 1),
                correct / max(total_samples, 1),
                bit_errors / max(total_bits, 1),
                current_batch_size,
            )
        except RuntimeError as error:
            if Config.DEVICE.type != "cuda" or not is_cuda_oom(error):
                raise
            if current_batch_size <= Config.MIN_BLOCK_SIZE:
                raise
            next_batch_size = max(current_batch_size // 2, Config.MIN_BLOCK_SIZE)
            log(f"eval | CUDA OOM at batch_size={current_batch_size}, retrying with {next_batch_size}")
            current_batch_size = next_batch_size
            torch.cuda.empty_cache()


@torch.inference_mode()
def compute_test_metrics(model: nn.Module, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    tx_cls = symbols_to_classes(data["tx_test"].to(Config.DEVICE))
    rx_cls = symbols_to_classes(data["rx_test"].to(Config.DEVICE))
    baseline_ber = calculate_ber_from_classes(tx_cls, rx_cls)
    eval_batch_size = data.get("eval_batch_size", Config.EVAL_BATCH_SIZE)
    test_loss, test_acc, test_ber, safe_eval_batch_size = evaluate_split(
        model, data["test_x"], data["test_y"], eval_batch_size
    )
    return {
        "baseline_ber": baseline_ber,
        "equalized_ber": test_ber,
        "accuracy": test_acc,
        "ser": 1.0 - test_acc,
        "test_loss": test_loss,
        "safe_eval_batch_size": safe_eval_batch_size,
        "improvement_abs": baseline_ber - test_ber,
        "improvement_rel": (1 - test_ber / baseline_ber) * 100 if baseline_ber > 0 else 0.0,
        "improvement_db": 10 * np.log10(baseline_ber / test_ber) if test_ber > 0 else float("inf"),
    }


def plot_results(history: Dict, eval_results: Dict, model_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    train_epochs = list(range(1, len(history["train_loss"]) + 1))
    val_epochs = history["val_epochs"]
    test_epochs = history["test_epochs"]

    axes[0, 0].plot(train_epochs, history["train_loss"], label="Train", linewidth=2, alpha=0.8)
    axes[0, 0].plot(val_epochs, history["val_loss"], label="Val", linewidth=2)
    if history["test_loss"]:
        axes[0, 0].plot(test_epochs, history["test_loss"], label="Test", linewidth=2, linestyle="--")
    axes[0, 0].set_title(f"{model_name} - Loss Curves", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(val_epochs, history["val_acc"], color="green", linewidth=2, label="Val Acc")
    if history["test_acc"]:
        axes[0, 1].plot(test_epochs, history["test_acc"], color="#0984e3", linewidth=2, linestyle="--", label="Test Acc")
    axes[0, 1].set_title(f"{model_name} - Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(train_epochs, history["lr"], color="red", linewidth=2)
    axes[0, 2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("LR")
    axes[0, 2].set_yscale("log")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(
        ["Baseline", f"{model_name} EQ"],
        [eval_results["baseline_ber"], eval_results["equalized_ber"]],
        color=["#ff7675", "#55efc4"],
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1, 0].set_title("BER Comparison (log scale)", fontweight="bold")
    axes[1, 0].set_ylabel("BER")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(axis="y", alpha=0.3)

    metrics = ["Abs Reduction\n(pp)", "Rel Improvement\n(%)", "SNR Gain\n(dB)"]
    values = [
        eval_results["improvement_abs"] * 100,
        eval_results["improvement_rel"],
        min(eval_results["improvement_db"], 20),
    ]
    axes[1, 1].bar(metrics, values, color=["#74b9ff", "#a29bfe", "#fd79a8"], edgecolor="black", linewidth=1.5)
    axes[1, 1].set_title("Improvement Metrics", fontweight="bold")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(axis="y", alpha=0.3)

    axes[1, 2].plot(train_epochs, [loss * 100 for loss in history["train_loss"]], label="Train Loss x100", alpha=0.7, linewidth=1)
    axes[1, 2].plot(val_epochs, [acc * 100 for acc in history["val_acc"]], label="Val Acc (%)", alpha=0.7, linewidth=2)
    if history["test_ber"]:
        axes[1, 2].plot(test_epochs, [ber * 100 for ber in history["test_ber"]], label="Test BER (%)", alpha=0.9, linewidth=2)
    axes[1, 2].set_title("Training Dynamics", fontweight="bold")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Value")
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].legend()

    plt.suptitle(f"{model_name} Equalizer Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_path = Config.OUT_DIR / f"ber_results_{model_name.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_summary(results: List[Dict[str, float]]):
    summary = pd.DataFrame(results).sort_values("equalized_ber").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(summary["model_type"], summary["equalized_ber"], color="#00b894", edgecolor="black")
    axes[0].set_title("Equalized BER by Architecture", fontweight="bold")
    axes[0].set_ylabel("BER")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(summary["model_type"], summary["improvement_rel"], color="#0984e3", edgecolor="black")
    axes[1].set_title("Relative BER Improvement", fontweight="bold")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Config.OUT_DIR / "architecture_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary.to_csv(Config.OUT_DIR / "architecture_comparison.csv", index=False)


def plot_sweep_per_model(df: pd.DataFrame, x_col: str, x_label: str, filename_prefix: str):
    for model_name in Config.MODEL_TYPES:
        model_df = df[df["model_type"] == model_name].sort_values(x_col)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(model_df[x_col], model_df["equalized_ber"], marker="o", linewidth=2)
        ax.set_title(f"{model_name.upper()} - BER vs {x_label}", fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("BER")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(Config.OUT_DIR / f"{filename_prefix}_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_sweep_overlay(df: pd.DataFrame, x_col: str, x_label: str, filename: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name in Config.MODEL_TYPES:
        model_df = df[df["model_type"] == model_name].sort_values(x_col)
        ax.plot(model_df[x_col], model_df["equalized_ber"], marker="o", linewidth=2, label=model_name.upper())
    ax.set_title(f"BER vs {x_label} - All Models", fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("BER")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_model_with_overrides(model_name: str, max_test_files: int, **overrides) -> Dict[str, float]:
    tracked_keys = [
        "CONTEXT_K",
        "SEQ_LEN",
        "HIDDEN_DIM",
        "LSTM_HIDDEN",
        "SAVE_BEST",
    ]
    previous = {key: getattr(Config, key) for key in tracked_keys}
    try:
        for key, value in overrides.items():
            setattr(Config, key, value)
        Config.SEQ_LEN = 2 * Config.CONTEXT_K + 1
        Config.SAVE_BEST = False
        data = prepare_data(max_test_files=max_test_files)
        model, _, results = train_one_model(model_name, data)
        if Config.DEVICE.type == "cuda":
            model.to("cpu")
            del model
            torch.cuda.empty_cache()
        return results
    finally:
        for key, value in previous.items():
            setattr(Config, key, value)


def run_sweep_experiments():
    log("\nRunning BER sweep experiments on one test file...")
    window_rows: List[Dict[str, float]] = []
    hidden_rows: List[Dict[str, float]] = []

    for model_name in Config.MODEL_TYPES:
        for context_k in Config.WINDOW_SWEEP_VALUES:
            log(f"sweep | {model_name} | window={context_k}")
            results = run_model_with_overrides(
                model_name,
                max_test_files=Config.SWEEP_TEST_FILES,
                CONTEXT_K=context_k,
            )
            window_rows.append(
                {
                    "model_type": model_name,
                    "context_k": context_k,
                    "seq_len": 2 * context_k + 1,
                    **results,
                }
            )

        for hidden_size in Config.HIDDEN_SWEEP_VALUES:
            log(f"sweep | {model_name} | hidden={hidden_size}")
            results = run_model_with_overrides(
                model_name,
                max_test_files=Config.SWEEP_TEST_FILES,
                HIDDEN_DIM=hidden_size,
                LSTM_HIDDEN=hidden_size,
            )
            hidden_rows.append(
                {
                    "model_type": model_name,
                    "hidden_size": hidden_size,
                    **results,
                }
            )

    window_df = pd.DataFrame(window_rows)
    hidden_df = pd.DataFrame(hidden_rows)
    window_df.to_csv(Config.OUT_DIR / "ber_vs_window.csv", index=False)
    hidden_df.to_csv(Config.OUT_DIR / "ber_vs_hidden.csv", index=False)

    plot_sweep_per_model(window_df, x_col="seq_len", x_label="Window Size", filename_prefix="ber_vs_window")
    plot_sweep_per_model(hidden_df, x_col="hidden_size", x_label="Hidden Size", filename_prefix="ber_vs_hidden")
    plot_sweep_overlay(window_df, x_col="seq_len", x_label="Window Size", filename="ber_vs_window_overlay.png")
    plot_sweep_overlay(hidden_df, x_col="hidden_size", x_label="Hidden Size", filename="ber_vs_hidden_overlay.png")


def train_one_model(model_name: str, data: Dict[str, torch.Tensor]) -> Tuple[nn.Module, Dict, Dict[str, float]]:
    model = make_model(model_name)
    if Config.USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=Config.TORCH_COMPILE_MODE)
            log(f"{model_name} | torch.compile enabled ({Config.TORCH_COMPILE_MODE})")
        except Exception as exc:
            log(f"{model_name} | torch.compile disabled: {exc}")

    optimizer_kwargs = {"lr": Config.LEARNING_RATE, "weight_decay": Config.WEIGHT_DECAY}
    if Config.DEVICE.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = None
    if Config.LR_SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            threshold=Config.SCHEDULER_THRESHOLD,
            min_lr=Config.MIN_LR,
        )

    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP)
    best_train_loss = float("inf")
    best_val_ber = float("inf")
    best_state = None
    best_test_metrics = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_ber": [],
        "val_epochs": [],
        "test_loss": [],
        "test_acc": [],
        "test_ber": [],
        "test_epochs": [],
        "lr": [],
        "epoch_time_sec": [],
        "train_samples_per_sec": [],
    }

    train_x = data["train_x"]
    train_y = data["train_y"]
    train_block_size = Config.TRAIN_BLOCK_SIZE
    eval_batch_size = Config.EVAL_BATCH_SIZE

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0
        total_train = train_x.size(0)
        num_blocks = (total_train + train_block_size - 1) // train_block_size
        block_order = torch.randperm(num_blocks).tolist()
        for block_idx in block_order:
            block_start = block_idx * train_block_size
            block_end = min(block_start + train_block_size, total_train)
            start = block_start
            while start < block_end:
                end = min(start + train_block_size, block_end)
                xb = train_x[start:end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
                yb = train_y[start:end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
                try:
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context():
                        mark_cudagraph_step_begin()
                        preds = model(xb)
                        loss = criterion(preds, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    batch_size_now = yb.size(0)
                    running_loss += loss.item() * batch_size_now
                    seen += batch_size_now
                    start = end
                except RuntimeError as error:
                    if Config.DEVICE.type != "cuda" or not is_cuda_oom(error):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    if train_block_size <= Config.MIN_BLOCK_SIZE:
                        raise
                    next_block_size = max(train_block_size // 2, Config.MIN_BLOCK_SIZE)
                    log(f"{model_name} | CUDA OOM at train_block_size={train_block_size}, retrying with {next_block_size}")
                    train_block_size = next_block_size
                    torch.cuda.empty_cache()

        train_loss = running_loss / max(seen, 1)
        epoch_time = time.time() - epoch_start
        speed = seen / max(epoch_time, 1e-9)

        val_loss, val_acc, val_ber, eval_batch_size = evaluate_split(model, data["val_x"], data["val_y"], eval_batch_size)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ber"].append(val_ber)
        history["val_epochs"].append(epoch + 1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time_sec"].append(epoch_time)
        history["train_samples_per_sec"].append(speed)

        if (epoch + 1) % Config.TEST_BER_EVERY == 0 or epoch == 0 or epoch == Config.EPOCHS - 1:
            test_metrics = compute_test_metrics(model, {**data, "eval_batch_size": eval_batch_size})
            eval_batch_size = int(test_metrics["safe_eval_batch_size"])
            history["test_loss"].append(test_metrics["test_loss"])
            history["test_acc"].append(test_metrics["accuracy"])
            history["test_ber"].append(test_metrics["equalized_ber"])
            history["test_epochs"].append(epoch + 1)
            best_test_metrics = test_metrics
            log(
                f"{model_name} | epoch {epoch+1:4d}/{Config.EPOCHS} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | val_ber {val_ber:.6e} | "
                f"test_ber {test_metrics['equalized_ber']:.6e} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"speed {speed:,.0f} samp/s | time {epoch_time:.1f}s"
            )
        elif epoch % Config.LOG_EVERY == 0:
            log(
                f"{model_name} | epoch {epoch+1:4d}/{Config.EPOCHS} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | val_ber {val_ber:.6e} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | speed {speed:,.0f} samp/s | time {epoch_time:.1f}s"
            )

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if val_ber < best_val_ber:
            best_val_ber = val_ber
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if Config.SAVE_BEST:
                torch.save(best_state, Config.OUT_DIR / f"{model_name}_best.pth")

        prev_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_ber)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                log(f"{model_name} | epoch {epoch+1} -- scheduler reduced lr to {new_lr:.6g}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_metrics = compute_test_metrics(model, {**data, "eval_batch_size": eval_batch_size})
    return model, history, final_metrics


def main():
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Device: {Config.DEVICE}")
    data = prepare_data()

    all_results = []
    for model_name in Config.MODEL_TYPES:
        log(f"\nTraining {model_name.upper()}...")
        model, history, results = train_one_model(model_name, data)
        results["model_type"] = model_name
        all_results.append(results)

        plot_results(history, results, model_name)
        torch.save(model.state_dict(), Config.OUT_DIR / f"{model_name}_final.pth")
        log(
            f"{model_name.upper()} | baseline BER {results['baseline_ber']:.6e} | "
            f"equalized BER {results['equalized_ber']:.6e} | acc {results['accuracy']:.4%} | "
            f"rel improvement {results['improvement_rel']:.2f}%"
        )

        if Config.DEVICE.type == "cuda":
            model.to("cpu")
            del model
            torch.cuda.empty_cache()

    plot_architecture_summary(all_results)
    log(f"Saved summary: {Config.OUT_DIR / 'architecture_comparison.csv'}")

    if Config.RUN_SWEEP_EXPERIMENTS:
        run_sweep_experiments()


if __name__ == "__main__":
    main()
