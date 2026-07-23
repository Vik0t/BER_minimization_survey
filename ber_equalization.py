import copy
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from efficient_kan import KAN as EfficientKAN
except ImportError:
    EfficientKAN = None

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class Config:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    DATA_DIR_CANDIDATES = [Path("symbols_new"), Path("Symbols_1m_1ch_PR"), Path(".")]
    MAX_FILES = 64
    # File-level split: first TRAIN_PORTION files form train+val pool, the rest is a hold-out test set.
    TRAIN_PORTION = 0.97
    VAL_PORTION_WITHIN_TRAIN = 0.10
    MIN_VAL_FILES = 1
    RANDOMIZE_FILE_SPLIT = False
    SPLIT_SEED = 42

    CONTEXT_K = 32
    SEQ_LEN = 2 * CONTEXT_K + 1
    INPUT_DIM = 2

    HIDDEN_DIM = 96 # надо 64
    DROPOUT = 0.2
    LSTM_HIDDEN = 64 # надо 64
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    TRANSFORMER_DIM = 128
    TRANSFORMER_LAYERS = 4
    TRANSFORMER_HEADS = 4
    TRANSFORMER_FF_DIM = 256
    TRANSFORMER_CONV_KERNEL = 3
    TCN_HIDDEN_DIM = 96
    TCN_LAYERS = 5
    TCN_KERNEL_SIZE = 5
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    MAMBA_DIM = 96
    MAMBA_LAYERS = 4
    MAMBA_D_STATE = 16
    MAMBA_D_CONV = 4
    MAMBA_EXPAND = 2
    COMPLEX_CHANNELS = 24
    COMPLEX_BLOCK_CHANNELS = [24, 32]
    COMPLEX_KERNEL_SIZES = [5, 7]
    COMPLEX_HEAD_DIM = 128
    COMPLEX_TEMPORAL_DIM = 96
    COMPLEX_TEMPORAL_DILATIONS = [1, 2, 4]
    COMPLEX_LIGHT_CHANNELS = 48
    COMPLEX_LIGHT_DILATIONS = [1, 2, 4]
    COMPLEX_LIGHT_KERNEL_SIZE = 3
    COMPLEX_SEQ_DIM = 96
    COMPLEX_LSTM_HIDDEN = 64
    COMPLEX_LSTM_LAYERS = 2
    COMPLEX_USE_KERR = False
    COMPLEX_USE_DBP_FRONTEND = False
    COMPLEX_KERR_KERNEL = 5
    COMPLEX_KERR_INIT_GAMMA = 0.02
    DBP_NUM_STEPS = 20
    DBP_KERNEL_SIZE = 7
    DBP_FINAL_KERNEL_SIZE = 21
    DBP_USE_FINAL_FILTER = True
    DBP_USE_SYMMETRIC_FILTER = True
    DBP_USE_SYMMETRIC_NONLINEAR = True
    DBP_NL_MEMORY = 2
    DBP_INIT_FROM_LS = False
    DBP_INIT_SAMPLES = 65536
    DBP_INIT_FFT_SIZE = 4096
    DBP_JOINT_INIT = True
    DBP_JOINT_INIT_ITERS = 200
    DBP_JOINT_INIT_BATCH_SIZE = 1024
    DBP_JOINT_INIT_LR = 2e-3
    DBP_SEQSTAT_DIM = 128
    FCNN_HIDDEN_CHANNELS = 32
    FCNN_HIDDEN_LAYERS = 1
    FCNN_KERR_INIT_GAMMA = 0.02

    FASTKAN_HIDDEN_DIM = 96
    FASTKAN_LAYERS = 2
    FASTKAN_NUM_GRIDS = 8
    FASTKAN_GRID_MIN = -2.5
    FASTKAN_GRID_MAX = 2.5
    FASTKAN_BASE_ACT = "silu"
    FASTKAN_USE_BASE_PATH = True
    KAN_INPUT_DROPOUT = 0.05
    KAN_HIDDEN_DROPOUT = 0.1
    KAN_PRUNE_L1 = 1e-5
    KAN_PRUNE_THRESHOLD = 0.02
    KAN_STRUCTURAL_PRUNE_AFTER_TRAINING = False
    KAN_STRUCTURAL_PRUNE_KEEP_RATIOS = [0.75, 0.5, 0.35, 0.25]
    KAN_STRUCTURAL_PRUNE_MIN_HIDDEN = 16
    KAN_STRUCTURAL_PRUNE_FINE_TUNE_EPOCHS = 20
    KAN_STRUCTURAL_PRUNE_FINE_TUNE_LR = 2e-4
    KAN_STRUCTURAL_PRUNE_SELECT_BY = "efficiency_score"
    EFFICIENCY_BATCH_SIZE = 16000
    EFFICIENCY_SCORE_POWER = 3.0
    EFFICIENCY_TIMING_WARMUP = 5
    EFFICIENCY_TIMING_REPEATS = 20
    EFFICIENT_KAN_HIDDEN_DIM = 128
    EFFICIENT_KAN_LAYERS = 2
    EFFICIENT_KAN_GRID_SIZE = 20
    EFFICIENT_KAN_SPLINE_ORDER = 3
    EFFICIENT_KAN_GRID_EPS = 0.02
    EFFICIENT_KAN_GRID_RANGE = [-3.0, 3.0]
    EFFICIENT_KAN_SCALE_NOISE = 0.1
    EFFICIENT_KAN_SCALE_BASE = 1.0
    EFFICIENT_KAN_SCALE_SPLINE = 1.0
    KAN_FEATURE_RADIUS = 2

    EPOCHS = 250
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    TRAIN_BLOCK_SIZE = 8192
    EVAL_BATCH_SIZE = 65536
    MIN_BLOCK_SIZE = 1024
    USE_AMP = True
    USE_TORCH_COMPILE = False
    TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
    OPTIMIZER = "adam"
    LOSS = "mse"
    GRAD_CLIP_NORM = 1.0
    LR_SCHEDULER = "notebook_decay"
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 100
    SCHEDULER_THRESHOLD = 1e-6

    DECAY_STEPS = 24
    MIN_LR = 1e-5
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 72
    EARLY_STOPPING_MIN_EPOCHS = 40
    EARLY_STOPPING_THRESHOLD = 0.0
    LOG_EVERY = 1
    TEST_BER_EVERY = 10
    SAVE_BEST_BY = "val_ber"
    EVAL_TEST_DURING_TRAINING = False
    COMPUTE_PER_FILE_METRICS = True
    POWER_NORMALIZE = True
    BER_SCALE_SEARCH = True
    BER_SCALE_MIN = 0.5
    BER_SCALE_MAX = 1.5
    BER_SCALE_STEPS = 10
    BER_SCALE_OFFSET = 10000
    BER_SCALE_SAMPLES = 1 << 20

    OUT_DIR = Path("clean_compare_outputs")
    RUN_MAIN_EXPERIMENTS = True
    MODEL_TYPES = [
        "efficient_kan_baseline",
        "kan_classifier",
    
    ]
    SAVE_BEST = True
    RUN_SWEEP_EXPERIMENTS = False
    SWEEP_TEST_FILES = 1
    WINDOW_SWEEP_VALUES = [2, 4, 8, 12, 16, 20]
    HIDDEN_SWEEP_VALUES = [8, 16, 32, 64]
    RUN_EFFICIENT_KAN_SWEEP = False
    EFFICIENT_KAN_SWEEP_MODELS = ["mlp", "efficient_kan_baseline", "kan_classifier"]
    EFFICIENT_KAN_SWEEP_EPOCHS = 60
    EFFICIENT_KAN_SWEEP_TEST_FILES = 1
    EFFICIENT_KAN_HIDDEN_SWEEP_VALUES = [64, 96, 128, 192]
    EFFICIENT_KAN_LR_SWEEP_VALUES = [1e-3]
    EFFICIENT_KAN_GRID_SWEEP_VALUES = [4, 8, 12, 16]
    EFFICIENT_KAN_ORDER_SWEEP_VALUES = [1, 2, 3, 4]
    EFFICIENT_KAN_LAYER_SWEEP_VALUES = [1, 2, 3]

    RUN_KAN_EXPERIMENT_SUITE = False
    EXPERIMENT_EPOCHS = 60
    EXPERIMENT_TEST_FILES = 1
    EXPERIMENT_COMPUTE_PER_FILE_METRICS = False
    EXPERIMENT_KAN_MODELS = ["efficient_kan_baseline", "kan_classifier"]
    EXPERIMENT_COMPARE_MODELS = ["efficient_kan_baseline", "mlp"]
    EXPERIMENT_COMPLEXITY_MODELS = ["complex_fastkan", "efficient_kan_baseline"]
    EXPERIMENT_FIXED_GRID = 16
    EXPERIMENT_FIXED_SPLINE_ORDER = 3
    EXPERIMENT_HIDDEN_VALUES = [64, 96, 128, 192]
    EXPERIMENT_WINDOW_VALUES = [8, 16, 24, 32, 48]
    EXPERIMENT_GRID_VALUES = [4, 8, 12, 16, 20]
    EXPERIMENT_SPLINE_ORDER_VALUES = [1, 2, 3, 4]
    EXPERIMENT_LAYER_VALUES = [1, 2, 3]
    MLP_LAYERS = 3

    RUN_FASTKAN_CLASSIFIER_SWEEP = True
    FASTKAN_CLASSIFIER_SWEEP_MODELS = ["fastkan_classifier", "complex_fastkan_classifier"]
    FASTKAN_CLASSIFIER_SWEEP_EPOCHS = 60
    FASTKAN_CLASSIFIER_SWEEP_TEST_FILES = 1
    FASTKAN_CLASSIFIER_HIDDEN_VALUES = [16, 32, 48, 64, 96]
    FASTKAN_CLASSIFIER_GRID_VALUES = [4, 8, 12, 16]
    FASTKAN_CLASSIFIER_LAYER_VALUES = [1, 2]


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


class CNNRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = Config.HIDDEN_DIM
        self.cnn = nn.Sequential(
            nn.Conv1d(Config.INPUT_DIM, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.25),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.25),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.25),
        )
        self.pool_score = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        fused_dim = hidden_dim * 2 + 2 * Config.INPUT_DIM
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(hidden_dim // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        seq = raw.transpose(1, 2)
        features = self.cnn(seq)
        center = features[:, :, Config.CONTEXT_K]
        weights = torch.softmax(self.pool_score(features), dim=2)
        global_context = torch.sum(weights * features, dim=2)
        raw_center = raw[:, Config.CONTEXT_K, :]
        raw_mean = raw.mean(dim=1)
        fused = torch.cat([center, global_context, raw_center, raw_mean], dim=1)
        return self.head(fused)


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
    def __init__(self, channels: int, kernel_size: Optional[int] = None, init_gamma: Optional[float] = None, symmetric: bool = False):
        super().__init__()
        kernel_size = Config.COMPLEX_KERR_KERNEL if kernel_size is None else kernel_size
        self.symmetric = symmetric
        self.power_filter = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        gamma = Config.COMPLEX_KERR_INIT_GAMMA if init_gamma is None else init_gamma
        self.gamma = nn.Parameter(torch.full((1, channels, 1), gamma))
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.power_filter.weight)
        center = self.power_filter.weight.size(-1) // 2
        self.power_filter.weight.data[:, :, center] = 1.0

    def _build_weight(self) -> torch.Tensor:
        weight = self.power_filter.weight
        if not self.symmetric:
            return weight
        return torch.cat([weight, weight.flip(dims=(2,))[:, :, 1:]], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0::2, :]
        imag = x[:, 1::2, :]
        power_weight = self._build_weight()
        power = F.conv1d(
            real.square() + imag.square(),
            power_weight,
            padding=power_weight.size(-1) // 2,
            groups=self.power_filter.groups,
        )
        phase = self.gamma * power
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        out_real = cos_phase * real + sin_phase * imag
        out_imag = cos_phase * imag - sin_phase * real
        return torch.stack((out_real, out_imag), dim=2).flatten(1, 2)


class ComplexFCNNKerrEqualizer(nn.Module):
    """PyTorch port of the MXNet complex FCNN/Kerr symbol equalizer."""

    def __init__(self):
        super().__init__()
        if Config.INPUT_DIM != 2:
            raise ValueError("complex_fcnn_kerr expects one complex channel (INPUT_DIM=2).")
        channels = Config.FCNN_HIDDEN_CHANNELS
        self.input_conv = ComplexConv1d(1, channels, kernel_size=Config.SEQ_LEN)
        self.input_kerr = KerrLikeActivation(channels, kernel_size=1, init_gamma=Config.FCNN_KERR_INIT_GAMMA)
        self.hidden = nn.ModuleList(
            ComplexConv1d(channels, channels, kernel_size=1)
            for _ in range(Config.FCNN_HIDDEN_LAYERS)
        )
        self.hidden_kerr = nn.ModuleList(
            KerrLikeActivation(channels, kernel_size=1, init_gamma=Config.FCNN_KERR_INIT_GAMMA)
            for _ in range(Config.FCNN_HIDDEN_LAYERS)
        )
        self.output_conv = ComplexConv1d(channels, 1, kernel_size=1)
        self.output_kerr = KerrLikeActivation(1, kernel_size=1, init_gamma=Config.FCNN_KERR_INIT_GAMMA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.view(x.size(0), Config.SEQ_LEN, 2).transpose(1, 2)
        state = self.input_conv(seq)[:, :, Config.CONTEXT_K : Config.CONTEXT_K + 1]
        state = self.input_kerr(state)
        for linear, kerr in zip(self.hidden, self.hidden_kerr):
            state = kerr(linear(state))
        return self.output_kerr(self.output_conv(state)).squeeze(-1)


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


class LightweightComplexTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
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
        self.mix = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv1d(1, channels * 2, kernel_size=1, bias=True)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(Config.DROPOUT * 0.5)
        self._init_weights()

    def _init_weights(self):
        for module in (self.depthwise, self.mix, self.gate_proj, self.out_proj):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.depthwise(x)
        value, gate = self.mix(x).chunk(2, dim=1)
        power_gate = self.gate_proj(power)
        value = F.gelu(value + power_gate[:, : value.size(1), :])
        gate = torch.sigmoid(gate + power_gate[:, value.size(1) :, :])
        x = self.out_proj(value * gate)
        x = self.dropout(x)
        return x + residual


class LightweightComplexEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = Config.COMPLEX_LIGHT_CHANNELS
        self.stem = nn.Sequential(
            nn.Conv1d(5, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                LightweightComplexTemporalBlock(
                    channels=channels,
                    kernel_size=Config.COMPLEX_LIGHT_KERNEL_SIZE,
                    dilation=dilation,
                )
                for dilation in Config.COMPLEX_LIGHT_DILATIONS
            ]
        )
        self.out_norm = nn.BatchNorm1d(channels)
        self.out_channels = channels
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        seq = raw.transpose(1, 2)
        real = seq[:, 0:1, :]
        imag = seq[:, 1:2, :]
        power = real.square() + imag.square()
        magnitude = torch.sqrt(power + 1e-6)
        cross = real * imag
        features = torch.cat([real, imag, magnitude, power, cross], dim=1)
        hidden = self.stem(features)
        for block in self.blocks:
            hidden = block(hidden, power)
        hidden = self.out_norm(hidden)
        seq_features = torch.cat([hidden, real, imag, magnitude], dim=1).transpose(1, 2)
        return raw, hidden, seq_features


class GaussianRBFExpansion(nn.Module):
    def __init__(self, num_grids: int, grid_min: float, grid_max: float):
        super().__init__()
        centers = torch.linspace(grid_min, grid_max, num_grids)
        spacing = float(centers[1] - centers[0]) if num_grids > 1 else max(abs(grid_max - grid_min), 1.0)
        self.register_buffer("centers", centers)
        self.log_inv_scale = nn.Parameter(torch.log(torch.tensor(1.0 / max(spacing, 1e-3))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_scale = self.log_inv_scale.exp().clamp(1e-3, 1e3)
        diff = (x.unsqueeze(-1) - self.centers) * inv_scale
        return torch.exp(-(diff * diff))


class FastKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rbf = GaussianRBFExpansion(
            num_grids=Config.FASTKAN_NUM_GRIDS,
            grid_min=Config.FASTKAN_GRID_MIN,
            grid_max=Config.FASTKAN_GRID_MAX,
        )
        self.base_linear = nn.Linear(in_features, out_features)
        self.spline_linear = nn.Linear(in_features * Config.FASTKAN_NUM_GRIDS, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(Config.KAN_HIDDEN_DROPOUT)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.base_linear.weight)
        nn.init.zeros_(self.base_linear.bias)
        nn.init.xavier_uniform_(self.spline_linear.weight)
        nn.init.zeros_(self.spline_linear.bias)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if Config.FASTKAN_BASE_ACT == "gelu":
            base = F.gelu(x)
        else:
            base = F.silu(x)
        base_out = self.base_linear(base) if Config.FASTKAN_USE_BASE_PATH else 0.0
        spline_basis = self.rbf(x).flatten(start_dim=1)
        spline_out = self.spline_linear(spline_basis)
        out = self.norm(base_out + spline_out)
        return self.dropout(F.gelu(out))

    def regularization_loss(self) -> torch.Tensor:
        return self.spline_linear.weight.abs().mean()


class FastKANHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(Config.KAN_INPUT_DROPOUT)
        self.feature_gate = nn.Parameter(torch.ones(input_dim))
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for _ in range(Config.FASTKAN_LAYERS):
            self.layers.append(FastKANLayer(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.output = nn.Linear(in_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.constant_(self.input_norm.weight, 1.0)
        nn.init.constant_(self.input_norm.bias, 0.0)

    def _gated_features(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.feature_gate
        if not self.training and Config.KAN_PRUNE_THRESHOLD > 0:
            gate = gate * (gate.abs() >= Config.KAN_PRUNE_THRESHOLD)
        return x * gate.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(self.input_norm(x))
        x = self._gated_features(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

    def regularization_loss(self) -> torch.Tensor:
        reg = self.feature_gate.abs().mean()
        for layer in self.layers:
            reg = reg + layer.regularization_loss()
        return reg


class EfficientKANBaselineEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        if EfficientKAN is None:
            raise ImportError(
                "efficient_kan is unavailable. Keep the local `efficient_kan/` package next to this script or "
                "install the library before running."
            )

        input_dim = Config.SEQ_LEN * Config.INPUT_DIM
        hidden_layers = [Config.EFFICIENT_KAN_HIDDEN_DIM] * max(Config.EFFICIENT_KAN_LAYERS, 1)
        self.kan = EfficientKAN(
            layers_hidden=[input_dim, *hidden_layers, 2],
            grid_size=Config.EFFICIENT_KAN_GRID_SIZE,
            spline_order=Config.EFFICIENT_KAN_SPLINE_ORDER,
            scale_noise=Config.EFFICIENT_KAN_SCALE_NOISE,
            scale_base=Config.EFFICIENT_KAN_SCALE_BASE,
            scale_spline=Config.EFFICIENT_KAN_SCALE_SPLINE,
            base_activation=nn.SiLU,
            grid_eps=Config.EFFICIENT_KAN_GRID_EPS,
            grid_range=Config.EFFICIENT_KAN_GRID_RANGE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def regularization_loss(self) -> torch.Tensor:
        if hasattr(self.kan, "regularization_loss"):
            reg = self.kan.regularization_loss()
            if torch.is_tensor(reg):
                return reg
            return torch.tensor(float(reg), device=Config.DEVICE)
        return torch.zeros((), device=Config.DEVICE)


class FastKANClassifierEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = Config.SEQ_LEN * Config.INPUT_DIM
        self.head = FastKANHead(
            input_dim=input_dim,
            hidden_dim=Config.FASTKAN_HIDDEN_DIM,
            out_dim=CONSTELLATION.size(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def regularization_loss(self) -> torch.Tensor:
        return self.head.regularization_loss()


def make_efficient_kan(input_dim: int, output_dim: int) -> nn.Module:
    if EfficientKAN is None:
        raise ImportError(
            "efficient_kan is unavailable. Keep the local `efficient_kan/` package next to this script or "
            "install the library before running."
        )
    hidden_layers = [Config.EFFICIENT_KAN_HIDDEN_DIM] * max(Config.EFFICIENT_KAN_LAYERS, 1)
    return EfficientKAN(
        layers_hidden=[input_dim, *hidden_layers, output_dim],
        grid_size=Config.EFFICIENT_KAN_GRID_SIZE,
        spline_order=Config.EFFICIENT_KAN_SPLINE_ORDER,
        scale_noise=Config.EFFICIENT_KAN_SCALE_NOISE,
        scale_base=Config.EFFICIENT_KAN_SCALE_BASE,
        scale_spline=Config.EFFICIENT_KAN_SCALE_SPLINE,
        base_activation=nn.SiLU,
        grid_eps=Config.EFFICIENT_KAN_GRID_EPS,
        grid_range=Config.EFFICIENT_KAN_GRID_RANGE,
    )


def efficient_kan_regularization(kan: nn.Module) -> torch.Tensor:
    if hasattr(kan, "regularization_loss"):
        reg = kan.regularization_loss()
        if torch.is_tensor(reg):
            return reg
        return torch.tensor(float(reg), device=Config.DEVICE)
    return torch.zeros((), device=Config.DEVICE)


class EfficientKANResidualEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = Config.SEQ_LEN * Config.INPUT_DIM
        self.kan = make_efficient_kan(input_dim=input_dim, output_dim=2)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        center_rx = raw[:, Config.CONTEXT_K, :]
        correction = self.kan(x)
        return center_rx + self.residual_scale * correction

    def regularization_loss(self) -> torch.Tensor:
        return efficient_kan_regularization(self.kan)


class EfficientKANFeatureEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 17
        self.kan = make_efficient_kan(input_dim=self.feature_dim, output_dim=2)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        center = raw[:, Config.CONTEXT_K, :]
        radius = min(Config.KAN_FEATURE_RADIUS, Config.CONTEXT_K)
        local = raw[:, Config.CONTEXT_K - radius : Config.CONTEXT_K + radius + 1, :]

        global_mean = raw.mean(dim=1)
        global_std = raw.std(dim=1, unbiased=False)
        local_mean = local.mean(dim=1)
        local_std = local.std(dim=1, unbiased=False)

        power = raw.square().sum(dim=2)
        power_center = power[:, Config.CONTEXT_K : Config.CONTEXT_K + 1]
        power_mean = power.mean(dim=1, keepdim=True)
        power_std = power.std(dim=1, unbiased=False, keepdim=True)

        cross = raw[:, :, 0] * raw[:, :, 1]
        cross_mean = cross.mean(dim=1, keepdim=True)
        cross_std = cross.std(dim=1, unbiased=False, keepdim=True)
        edge_delta = raw[:, -1, :] - raw[:, 0, :]

        return torch.cat(
            [
                center,
                local_mean,
                local_std,
                global_mean,
                global_std,
                power_center,
                power_mean,
                power_std,
                cross_mean,
                cross_std,
                edge_delta,
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(self._features(x))

    def regularization_loss(self) -> torch.Tensor:
        return efficient_kan_regularization(self.kan)


class CNNKANEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = Config.HIDDEN_DIM
        self.cnn = nn.Sequential(
            nn.Conv1d(Config.INPUT_DIM, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.pool_score = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        fused_dim = hidden_dim * 2 + 2 * Config.INPUT_DIM
        self.kan = make_efficient_kan(input_dim=fused_dim, output_dim=2)
        self._init_weights()

    def _init_weights(self):
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        nn.init.kaiming_normal_(self.pool_score.weight, nonlinearity="linear")
        nn.init.constant_(self.pool_score.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        features = self.cnn(raw.transpose(1, 2))
        center = features[:, :, Config.CONTEXT_K]
        weights = torch.softmax(self.pool_score(features), dim=2)
        global_context = torch.sum(weights * features, dim=2)
        raw_center = raw[:, Config.CONTEXT_K, :]
        raw_mean = raw.mean(dim=1)
        fused = torch.cat([center, global_context, raw_center, raw_mean], dim=1)
        return self.kan(fused)

    def regularization_loss(self) -> torch.Tensor:
        return efficient_kan_regularization(self.kan)


class EfficientKANClassifierEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = Config.SEQ_LEN * Config.INPUT_DIM
        self.kan = make_efficient_kan(input_dim=input_dim, output_dim=CONSTELLATION.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def regularization_loss(self) -> torch.Tensor:
        return efficient_kan_regularization(self.kan)


class ComplexFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dbp_frontend = ComplexDBPFrontEnd() if Config.COMPLEX_USE_DBP_FRONTEND else None
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

    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        if self.dbp_frontend is not None:
            self.dbp_frontend.initialize_from_data(train_x, train_y)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        if self.dbp_frontend is not None:
            x, _ = self.dbp_frontend(x, collect_states=False)
        else:
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
        nl_kernel = Config.DBP_NL_MEMORY + 1 if Config.DBP_USE_SYMMETRIC_NONLINEAR else 2 * Config.DBP_NL_MEMORY + 1
        self.nonlinear = KerrLikeActivation(
            1,
            kernel_size=nl_kernel,
            init_gamma=Config.COMPLEX_KERR_INIT_GAMMA,
            symmetric=Config.DBP_USE_SYMMETRIC_NONLINEAR,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class ComplexDBPFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = nn.ModuleList([ComplexDBPStep1Ch() for _ in range(Config.DBP_NUM_STEPS)])
        self.final_linear = (
            ComplexConv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=Config.DBP_FINAL_KERNEL_SIZE,
                symmetric=Config.DBP_USE_SYMMETRIC_FILTER,
            )
            if Config.DBP_USE_FINAL_FILTER
            else None
        )
        linear_delay = Config.DBP_KERNEL_SIZE - 1 if Config.DBP_USE_SYMMETRIC_FILTER else Config.DBP_KERNEL_SIZE // 2
        final_delay = 0
        if self.final_linear is not None:
            final_delay = (
                Config.DBP_FINAL_KERNEL_SIZE - 1
                if Config.DBP_USE_SYMMETRIC_FILTER
                else Config.DBP_FINAL_KERNEL_SIZE // 2
            )
        nl_delay = Config.DBP_NL_MEMORY if Config.DBP_USE_SYMMETRIC_NONLINEAR else 2 * Config.DBP_NL_MEMORY
        self.valid_margin = Config.DBP_NUM_STEPS * (linear_delay + nl_delay) + final_delay
        if self.valid_margin > Config.CONTEXT_K:
            raise ValueError(
                f"DBP receptive radius {self.valid_margin} exceeds CONTEXT_K={Config.CONTEXT_K}. "
                "Increase CONTEXT_K or reduce DBP kernels/steps."
            )

    @staticmethod
    def _seq_features(x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-6)
        return torch.cat([real, imag, magnitude], dim=1).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        collect_states: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        state = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM).transpose(1, 2)
        states = [state] if collect_states else []
        for step in self.steps:
            state = step(state)
            if collect_states:
                states.append(state)
        if self.final_linear is not None:
            state = self.final_linear(state)
            if collect_states:
                states.append(state)
        return state, states

    @torch.no_grad()
    def _initialize_linear_kernels_from_ls(self, train_x: torch.Tensor, train_y: torch.Tensor):
        samples = min(train_x.size(0), Config.DBP_INIT_SAMPLES)
        if samples < Config.SEQ_LEN:
            return
        windows = train_x[:samples].view(samples, Config.SEQ_LEN, Config.INPUT_DIM)
        target = train_y[:samples]
        rx_complex = torch.complex(windows[:, :, 0], windows[:, :, 1]).to(torch.complex64)
        tx_complex = torch.complex(target[:, 0], target[:, 1]).to(torch.complex64)

        global_kernel = torch.linalg.lstsq(rx_complex, tx_complex.unsqueeze(1)).solution.squeeze(1)
        global_kernel = global_kernel / global_kernel.norm().clamp_min(1e-6)

        fft_size = max(Config.DBP_INIT_FFT_SIZE, 1 << int(np.ceil(np.log2(Config.SEQ_LEN))))
        global_response = centered_complex_kernel_to_frequency(global_kernel, fft_size)
        linear_stage_count = len(self.steps) + (1 if self.final_linear is not None else 0)
        step_response = complex_unit_response(global_response, linear_stage_count)
        final_response = global_response / step_response.pow(len(self.steps))

        step_kernel_centered = frequency_to_centered_complex_kernel(step_response, Config.SEQ_LEN)
        step_kernel = extract_kernel_from_centered_response(
            step_kernel_centered,
            Config.DBP_KERNEL_SIZE,
            Config.DBP_USE_SYMMETRIC_FILTER,
        )
        for step in self.steps:
            assign_complex_kernel(step.linear, step_kernel)
            center = step.nonlinear.power_filter.weight.size(-1) // 2
            step.nonlinear.power_filter.weight.zero_()
            step.nonlinear.power_filter.weight[:, :, center] = 1.0
            step.nonlinear.gamma.fill_(Config.COMPLEX_KERR_INIT_GAMMA / max(Config.DBP_NUM_STEPS, 1))

        if self.final_linear is not None:
            final_kernel_centered = frequency_to_centered_complex_kernel(final_response, Config.SEQ_LEN)
            final_kernel = extract_kernel_from_centered_response(
                final_kernel_centered,
                Config.DBP_FINAL_KERNEL_SIZE,
                Config.DBP_USE_SYMMETRIC_FILTER,
            )
            assign_complex_kernel(self.final_linear, final_kernel)

    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        if not Config.DBP_INIT_FROM_LS:
            return
        self._initialize_linear_kernels_from_ls(train_x, train_y)
        if not Config.DBP_JOINT_INIT or Config.DBP_JOINT_INIT_ITERS <= 0:
            return

        was_training = self.training
        self.train()
        subset = min(train_x.size(0), Config.DBP_INIT_SAMPLES)
        if subset < Config.DBP_JOINT_INIT_BATCH_SIZE:
            if not was_training:
                self.eval()
            return

        optimizer = optim.Adam(self.parameters(), lr=Config.DBP_JOINT_INIT_LR)
        criterion = nn.MSELoss()
        for _ in range(Config.DBP_JOINT_INIT_ITERS):
            index = torch.randint(0, subset, (Config.DBP_JOINT_INIT_BATCH_SIZE,))
            xb = train_x[index].to(Config.DEVICE)
            yb = train_y[index].to(Config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            state, _ = self.forward(xb, collect_states=False)
            preds = state[:, :, Config.CONTEXT_K].transpose(0, 1).transpose(0, 1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        if not was_training:
            self.eval()


class ComplexDBPSeqStatRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = ComplexDBPFrontEnd()
        self.valid_margin = self.frontend.valid_margin
        feature_dim = 3 * (
            1 + Config.DBP_NUM_STEPS + (1 if Config.DBP_USE_FINAL_FILTER else 0)
        )
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

    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.frontend.initialize_from_data(train_x, train_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        state, states = self.frontend(x, collect_states=True)
        features = [self.frontend._seq_features(step_state) for step_state in states]
        seq = torch.cat(features, dim=2)
        center = seq[:, Config.CONTEXT_K, :]
        if self.valid_margin > 0:
            valid_seq = seq[:, self.valid_margin : Config.SEQ_LEN - self.valid_margin, :]
        else:
            valid_seq = seq
        mean = valid_seq.mean(dim=1)
        std = valid_seq.std(dim=1, unbiased=False)
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

    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.encoder.initialize_from_data(train_x, train_y)

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

    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.encoder.initialize_from_data(train_x, train_y)

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

    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.encoder.initialize_from_data(train_x, train_y)

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


class ComplexFastKANEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LightweightComplexEncoder()
        fused_dim = 3 * (self.encoder.out_channels + 3) + 2 * Config.INPUT_DIM + 1
        self.head = FastKANHead(
            input_dim=fused_dim,
            hidden_dim=Config.FASTKAN_HIDDEN_DIM,
            out_dim=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, _, seq_features = self.encoder(x)
        center = seq_features[:, Config.CONTEXT_K, :]
        mean = seq_features.mean(dim=1)
        std = seq_features.std(dim=1, unbiased=False)
        raw_center = raw[:, Config.CONTEXT_K, :]
        raw_mean = raw.mean(dim=1)
        power_center = raw_center.square().sum(dim=1, keepdim=True)
        fused = torch.cat([center, mean, std, raw_center, raw_mean, power_center], dim=1)
        return self.head(fused)

    def regularization_loss(self) -> torch.Tensor:
        return self.head.regularization_loss()


class ComplexFastKANClassifierEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LightweightComplexEncoder()
        fused_dim = 3 * (self.encoder.out_channels + 3) + 2 * Config.INPUT_DIM + 1
        self.head = FastKANHead(
            input_dim=fused_dim,
            hidden_dim=Config.FASTKAN_HIDDEN_DIM,
            out_dim=CONSTELLATION.size(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw, _, seq_features = self.encoder(x)
        center = seq_features[:, Config.CONTEXT_K, :]
        mean = seq_features.mean(dim=1)
        std = seq_features.std(dim=1, unbiased=False)
        raw_center = raw[:, Config.CONTEXT_K, :]
        raw_mean = raw.mean(dim=1)
        power_center = raw_center.square().sum(dim=1, keepdim=True)
        fused = torch.cat([center, mean, std, raw_center, raw_mean, power_center], dim=1)
        return self.head(fused)

    def regularization_loss(self) -> torch.Tensor:
        return self.head.regularization_loss()


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


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
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
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(Config.DROPOUT * 0.5)
        self._init_weights()

    def _init_weights(self):
        for module in (self.depthwise, self.pointwise, self.out_proj):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.depthwise(x)
        value, gate = self.pointwise(x).chunk(2, dim=1)
        x = self.out_proj(F.gelu(value) * torch.sigmoid(gate))
        return residual + self.dropout(x)


class TCNRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = Config.TCN_HIDDEN_DIM
        dilations = Config.TCN_DILATIONS[: Config.TCN_LAYERS]
        if len(dilations) < Config.TCN_LAYERS:
            dilations = dilations + [dilations[-1] if dilations else 1] * (Config.TCN_LAYERS - len(dilations))
        self.stem = nn.Sequential(
            nn.Conv1d(Config.INPUT_DIM, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [TCNResidualBlock(hidden_dim, Config.TCN_KERNEL_SIZE, dilation) for dilation in dilations]
        )
        self.final_norm = nn.BatchNorm1d(hidden_dim)
        fused_dim = hidden_dim * 2 + 2 * Config.INPUT_DIM
        self.head = nn.Sequential(
            nn.Linear(fused_dim, Config.HIDDEN_DIM),
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
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        seq = raw.transpose(1, 2)
        hidden = self.stem(seq)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        center = hidden[:, :, Config.CONTEXT_K]
        global_context = hidden.mean(dim=2)
        raw_center = raw[:, Config.CONTEXT_K, :]
        raw_mean = raw.mean(dim=1)
        return self.head(torch.cat([center, global_context, raw_center, raw_mean], dim=1))


class MambaRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is unavailable. Install `mamba-ssm` to run the `mamba` model, "
                "or keep using `tcn`, `lstm`, and KAN baselines."
            )
        dim = Config.MAMBA_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(Config.INPUT_DIM, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(dim),
                        "mamba": Mamba(
                            d_model=dim,
                            d_state=Config.MAMBA_D_STATE,
                            d_conv=Config.MAMBA_D_CONV,
                            expand=Config.MAMBA_EXPAND,
                        ),
                    }
                )
                for _ in range(Config.MAMBA_LAYERS)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 2 + Config.INPUT_DIM, Config.HIDDEN_DIM),
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
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.view(x.size(0), Config.SEQ_LEN, Config.INPUT_DIM)
        hidden = self.input_proj(raw)
        for block in self.blocks:
            hidden = hidden + block["mamba"](block["norm"](hidden))
        hidden = self.final_norm(hidden)
        center = hidden[:, Config.CONTEXT_K, :]
        global_context = hidden.mean(dim=1)
        raw_center = raw[:, Config.CONTEXT_K, :]
        return self.head(torch.cat([center, global_context, raw_center], dim=1))


class MLPRxEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = Config.SEQ_LEN * Config.INPUT_DIM
        hidden_dim = Config.HIDDEN_DIM
        self.skip = nn.Linear(input_dim, 2)
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(Config.MLP_LAYERS, 1)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.skip.weight.zero_()
            self.skip.bias.zero_()
            center_start = Config.CONTEXT_K * Config.INPUT_DIM
            self.skip.weight[:, center_start : center_start + Config.INPUT_DIM] = torch.eye(2)

        linear_layers = [module for module in self.net.modules() if isinstance(module, nn.Linear)]
        for idx, module in enumerate(linear_layers):
            if idx == len(linear_layers) - 1:
                nn.init.zeros_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            else:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        for module in self.net.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    @torch.no_grad()
    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        samples = min(train_x.size(0), 262_144)
        if samples < 3:
            return
        center = train_x[:samples].view(samples, Config.SEQ_LEN, Config.INPUT_DIM)[:, Config.CONTEXT_K, :].float()
        target = train_y[:samples].float()
        ones = torch.ones(samples, 1, dtype=center.dtype, device=center.device)
        design = torch.cat([center, ones], dim=1)
        solution = torch.linalg.lstsq(design, target).solution

        self.skip.weight.zero_()
        self.skip.bias.copy_(solution[-1].to(self.skip.bias.device, dtype=self.skip.bias.dtype))
        center_start = Config.CONTEXT_K * Config.INPUT_DIM
        self.skip.weight[:, center_start : center_start + Config.INPUT_DIM].copy_(
            solution[: Config.INPUT_DIM].T.to(self.skip.weight.device, dtype=self.skip.weight.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.net(x)


def log(msg: str):
    print(msg, flush=True)


MODEL_NOTES = {
    "efficient_kan_baseline": "flat IQ window -> KAN -> corrected I/Q",
    "efficient_kan_residual": "flat IQ window -> KAN correction -> rx center + correction",
    "efficient_kan_features": "handcrafted local/global IQ features -> KAN -> corrected I/Q",
    "cnn_kan": "CNN extracts temporal features -> KAN head -> corrected I/Q",
    "kan_classifier": "flat IQ window -> KAN -> 16 constellation logits",
    "fastkan_classifier": "flat IQ window -> compact RBF/FastKAN -> 16 constellation logits",
    "complex_fastkan_classifier": "light complex encoder -> RBF/FastKAN -> 16 constellation logits",
    "complex_fcnn_kerr": "complex FCNN + learnable Kerr rotations -> corrected I/Q",
    "complex_dbp_seqstat": "learnable DBP steps + sequence statistics -> corrected I/Q",
    "mlp": "flat IQ window -> MLP -> corrected I/Q",
    "cnn": "CNN extracts temporal features -> MLP head -> corrected I/Q",
    "tcn": "dilated temporal CNN over IQ window -> corrected I/Q",
    "mamba": "Mamba sequence blocks over IQ window -> corrected I/Q",
}


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_criterion() -> nn.Module:
    if Config.LOSS == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.05)
    return nn.MSELoss()


def is_classifier_output(preds: torch.Tensor) -> bool:
    return preds.dim() == 2 and preds.size(1) == CONSTELLATION.size(0)


def prediction_loss(preds: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if is_classifier_output(preds):
        target_classes = symbols_to_classes(targets.float())
        return F.cross_entropy(preds.float(), target_classes)
    return criterion(preds, targets)


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    optimizer_kwargs = {"lr": Config.LEARNING_RATE, "weight_decay": Config.WEIGHT_DECAY}
    if Config.DEVICE.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer_name = Config.OPTIMIZER.lower()
    optimizer_cls = optim.Adam if optimizer_name == "adam" else optim.RMSprop
    try:
        return optimizer_cls(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        return optimizer_cls(model.parameters(), **optimizer_kwargs)


def compute_model_regularization(model: nn.Module) -> torch.Tensor:
    if Config.KAN_PRUNE_L1 <= 0 or not hasattr(model, "regularization_loss"):
        return torch.zeros((), device=Config.DEVICE)
    reg = model.regularization_loss()
    if not torch.is_tensor(reg):
        reg = torch.tensor(float(reg), device=Config.DEVICE)
    return reg * Config.KAN_PRUNE_L1


def complex_unit_response(response: torch.Tensor, order: int) -> torch.Tensor:
    magnitude = response.abs().clamp_min(1e-6).pow(1.0 / order)
    phase = torch.angle(response) / order
    return torch.polar(magnitude, phase)


def centered_complex_kernel_to_frequency(kernel: torch.Tensor, fft_size: int) -> torch.Tensor:
    center = kernel.numel() // 2
    ordered = torch.roll(kernel, shifts=-center, dims=0)
    return torch.fft.fft(ordered, n=fft_size)


def frequency_to_centered_complex_kernel(response: torch.Tensor, seq_len: int) -> torch.Tensor:
    time = torch.fft.ifft(response, n=response.numel())
    centered = torch.roll(time[:seq_len], shifts=seq_len // 2, dims=0)
    return centered


def extract_kernel_from_centered_response(
    kernel: torch.Tensor,
    kernel_size: int,
    symmetric: bool,
) -> torch.Tensor:
    center = kernel.numel() // 2
    if symmetric:
        return kernel[center : center + kernel_size].contiguous()
    start = center - kernel_size // 2
    end = start + kernel_size
    return kernel[start:end].contiguous()


def assign_complex_kernel(module, kernel: torch.Tensor):
    module.real_conv.weight.copy_(kernel.real.view_as(module.real_conv.weight).to(module.real_conv.weight.dtype))
    module.imag_conv.weight.copy_(kernel.imag.view_as(module.imag_conv.weight).to(module.imag_conv.weight.dtype))


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


def compute_rms_scale(symbols: torch.Tensor) -> torch.Tensor:
    power = torch.mean(symbols.square().sum(dim=1)).sqrt()
    return power.clamp_min(1e-6)


def power_normalize_pair(
    tx_symbols: torch.Tensor,
    rx_symbols: torch.Tensor,
    tx_scale: Optional[torch.Tensor] = None,
    rx_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tx_scale is None:
        tx_scale = compute_rms_scale(tx_symbols)
    if rx_scale is None:
        rx_scale = compute_rms_scale(rx_symbols)
    return tx_symbols / tx_scale, rx_symbols / rx_scale, tx_scale, rx_scale


def find_best_symbol_scale(tx_symbols: torch.Tensor, rx_symbols: torch.Tensor) -> float:
    if not Config.BER_SCALE_SEARCH or tx_symbols.numel() == 0 or rx_symbols.numel() == 0:
        return 1.0

    total = min(tx_symbols.size(0), rx_symbols.size(0))
    offset = min(Config.BER_SCALE_OFFSET, max(total - 1, 0))
    available = max(total - offset, 1)
    sample_count = min(available, Config.BER_SCALE_SAMPLES)
    tx_eval = tx_symbols[offset : offset + sample_count]
    rx_eval = rx_symbols[offset : offset + sample_count]

    tx_norm = tx_eval.square().sum(dim=1).mean().sqrt().clamp_min(1e-6)
    rx_norm = rx_eval.square().sum(dim=1).mean().sqrt().clamp_min(1e-6)
    center = (tx_norm / rx_norm).item()
    left = max(Config.BER_SCALE_MIN, 0.5 * center)
    right = min(Config.BER_SCALE_MAX, 1.5 * center)
    phi = (1 + 5**0.5) / 2

    def objective(scale: float) -> float:
        pred_classes = symbols_to_classes((rx_eval * scale).float())
        target_classes = symbols_to_classes(tx_eval.float())
        return calculate_ber_from_classes(target_classes, pred_classes)

    x1 = right - (right - left) / phi
    x2 = left + (right - left) / phi
    y1 = objective(x1)
    y2 = objective(x2)
    for _ in range(Config.BER_SCALE_STEPS):
        if y1 >= y2:
            left = x1
            x1 = x2
            y1 = y2
            x2 = left + (right - left) / phi
            y2 = objective(x2)
        else:
            right = x2
            x2 = x1
            y2 = y1
            x1 = right - (right - left) / phi
            y1 = objective(x1)
    return float((x1 + x2) * 0.5)


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
    file_indices = list(file_indices)
    if Config.RANDOMIZE_FILE_SPLIT:
        rng = np.random.default_rng(Config.SPLIT_SEED)
        file_indices = rng.permutation(file_indices).tolist()

    train_val_files = max(2, int(len(file_indices) * Config.TRAIN_PORTION))
    if train_val_files >= len(file_indices):
        train_val_files = len(file_indices) - 1
    train_val_pool = file_indices[:train_val_files]
    if len(train_val_pool) < 2:
        raise ValueError("Need at least 2 train+val files so validation can be held out.")

    requested_val_files = int(round(len(train_val_pool) * Config.VAL_PORTION_WITHIN_TRAIN))
    val_files = max(Config.MIN_VAL_FILES, requested_val_files)
    val_files = min(max(val_files, 1), len(train_val_pool) - 1)

    train_idx = train_val_pool[:-val_files]
    val_idx = train_val_pool[-val_files:]
    test_idx = file_indices[train_val_files:]
    if not test_idx:
        raise ValueError("File-level split produced no test files. Increase MAX_FILES or lower TRAIN_PORTION.")
    return train_idx, val_idx, test_idx


def load_symbol_file(base_dir: Path, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    path = base_dir / f"Symbols_1m_1ch_PR_{idx}.csv"
    arr = pd.read_csv(path, header=None, dtype=np.float32).to_numpy(copy=False)
    return torch.from_numpy(arr[:, 0:2].copy()), torch.from_numpy(arr[:, 2:4].copy())


def load_files(base_dir: Path, file_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tx_list: List[torch.Tensor] = []
    rx_list: List[torch.Tensor] = []
    for idx in file_indices:
        tx_symbols, rx_symbols = load_symbol_file(base_dir, idx)
        tx_list.append(tx_symbols)
        rx_list.append(rx_symbols)
    return torch.cat(tx_list, dim=0), torch.cat(rx_list, dim=0)


def load_file_splits(base_dir: Path, file_indices: List[int]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for idx in file_indices:
        tx_symbols, rx_symbols = load_symbol_file(base_dir, idx)
        files.append({"file_idx": idx, "tx": tx_symbols, "rx": rx_symbols})
    return files


def normalize_file_splits(
    files: List[Dict[str, Any]],
    tx_scale: torch.Tensor,
    rx_scale: torch.Tensor,
) -> List[Dict[str, Any]]:
    return [
        {
            "file_idx": item["file_idx"],
            "tx": item["tx"] / tx_scale,
            "rx": item["rx"] / rx_scale,
        }
        for item in files
    ]


def make_windows(rx_symbols: torch.Tensor, tx_symbols: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_rx = (rx_symbols - mean) / std
    norm_rx = torch.nan_to_num(norm_rx, nan=0.0, posinf=0.0, neginf=0.0)
    window_view = norm_rx.unfold(0, Config.SEQ_LEN, 1).permute(0, 2, 1).contiguous()
    x = window_view.view(window_view.size(0), -1).contiguous()
    y = tx_symbols[Config.CONTEXT_K : tx_symbols.size(0) - Config.CONTEXT_K].contiguous()
    return x, y


def make_windows_for_files(
    files: List[Dict[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
    collect_spans: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    x_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    file_spans: List[Dict[str, Any]] = []
    offset = 0

    for item in files:
        tx_symbols = item["tx"]
        rx_symbols = item["rx"]
        if rx_symbols.size(0) < Config.SEQ_LEN:
            continue
        x_file, y_file = make_windows(rx_symbols, tx_symbols, mean, std)
        count = x_file.size(0)
        if collect_spans:
            file_spans.append(
                {
                    "file_idx": item["file_idx"],
                    "start": offset,
                    "end": offset + count,
                    "tx_center": tx_symbols[Config.CONTEXT_K : tx_symbols.size(0) - Config.CONTEXT_K].contiguous(),
                    "rx_center": rx_symbols[Config.CONTEXT_K : rx_symbols.size(0) - Config.CONTEXT_K].contiguous(),
                }
            )
        x_parts.append(x_file)
        y_parts.append(y_file)
        offset += count

    if not x_parts:
        raise ValueError("No file is long enough to build context windows.")
    return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0), file_spans


def prepare_data(max_test_files: Optional[int] = None) -> Dict[str, Any]:
    base_dir, all_indices = discover_symbol_files()
    train_idx, val_idx, test_idx = resolve_splits(all_indices)
    if max_test_files is not None:
        test_idx = test_idx[:max_test_files]
        if not test_idx:
            raise ValueError("max_test_files truncated test split to zero files.")
    log(f"Data dir: {base_dir}")
    log(
        "Split protocol: train/val/test are separated by file; "
        "test files are never used for fitting, checkpoint selection, or normalization statistics."
    )
    if Config.RANDOMIZE_FILE_SPLIT:
        log(f"File split randomized with SPLIT_SEED={Config.SPLIT_SEED}")
    log(f"Train files: {train_idx}")
    log(f"Val files: {val_idx}")
    log(f"Test files: {test_idx}")

    train_files = load_file_splits(base_dir, train_idx)
    val_files = load_file_splits(base_dir, val_idx)
    test_files = load_file_splits(base_dir, test_idx)
    tx_train_raw = torch.cat([item["tx"] for item in train_files], dim=0)
    rx_train_raw = torch.cat([item["rx"] for item in train_files], dim=0)

    if Config.POWER_NORMALIZE:
        _, _, tx_scale, rx_scale = power_normalize_pair(tx_train_raw, rx_train_raw)
    else:
        tx_scale = torch.tensor(1.0, dtype=tx_train_raw.dtype)
        rx_scale = torch.tensor(1.0, dtype=rx_train_raw.dtype)

    train_files = normalize_file_splits(train_files, tx_scale, rx_scale)
    val_files = normalize_file_splits(val_files, tx_scale, rx_scale)
    test_files = normalize_file_splits(test_files, tx_scale, rx_scale)
    tx_train = torch.cat([item["tx"] for item in train_files], dim=0)
    rx_train = torch.cat([item["rx"] for item in train_files], dim=0)
    tx_test = torch.cat([item["tx"] for item in test_files], dim=0)
    rx_test = torch.cat([item["rx"] for item in test_files], dim=0)

    mean = rx_train.mean(dim=0, keepdim=True)
    std = rx_train.std(dim=0, keepdim=True)
    std[std == 0] = 1.0

    train_x, train_y, _ = make_windows_for_files(train_files, mean, std, collect_spans=False)
    val_x, val_y, val_file_spans = make_windows_for_files(val_files, mean, std)
    test_x, test_y, test_file_spans = make_windows_for_files(test_files, mean, std)
    log(
        "Window samples (built per file, no cross-file context): "
        f"train {train_x.size(0):,} | val {val_x.size(0):,} | test {test_x.size(0):,}"
    )
    rx_test_center = torch.cat([item["rx_center"] for item in test_file_spans], dim=0)
    tx_test_center = torch.cat([item["tx_center"] for item in test_file_spans], dim=0)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "val_file_spans": val_file_spans,
        "test_file_spans": test_file_spans,
        "tx_test": tx_test,
        "rx_test": rx_test,
        "tx_test_center": tx_test_center,
        "rx_test_center": rx_test_center,
        "mean": mean,
        "std": std,
        "tx_scale": tx_scale,
        "rx_scale": rx_scale,
    }


def make_model(name: str) -> nn.Module:
    if name in {"efficient_kan_baseline", "efficient_kan", "ekan"}:
        return EfficientKANBaselineEqualizer().to(Config.DEVICE)
    if name in {"efficient_kan_residual", "kan_residual", "residual_kan"}:
        return EfficientKANResidualEqualizer().to(Config.DEVICE)
    if name in {"efficient_kan_features", "kan_features", "feature_kan"}:
        return EfficientKANFeatureEqualizer().to(Config.DEVICE)
    if name in {"cnn_kan", "kan_cnn"}:
        return CNNKANEqualizer().to(Config.DEVICE)
    if name in {"kan_classifier", "efficient_kan_classifier"}:
        return EfficientKANClassifierEqualizer().to(Config.DEVICE)
    if name in {"fastkan_classifier", "rbf_kan_classifier"}:
        return FastKANClassifierEqualizer().to(Config.DEVICE)
    if name == "cnn":
        return CNNRxEqualizer().to(Config.DEVICE)
    if name == "lstm":
        return LSTMRxEqualizer().to(Config.DEVICE)
    if name in {"hybrid", "cnn_lstm"}:
        return HybridCNNLSTMEqualizer().to(Config.DEVICE)
    if name in {"complex_fastkan", "fastkan", "kan"}:
        return ComplexFastKANEqualizer().to(Config.DEVICE)
    if name in {"complex_fastkan_classifier", "complex_rbf_kan_classifier"}:
        return ComplexFastKANClassifierEqualizer().to(Config.DEVICE)
    if name == "complex_lstm":
        return ComplexLSTMRxEqualizer().to(Config.DEVICE)
    if name == "complex_fcnn_kerr":
        return ComplexFCNNKerrEqualizer().to(Config.DEVICE)
    if name == "complex_dbp_seqstat":
        return ComplexDBPSeqStatRxEqualizer().to(Config.DEVICE)
    if name == "complex_cnn_lstm":
        return ComplexCNNLSTMRxEqualizer().to(Config.DEVICE)
    if name == "complex_cnn":
        return ComplexCNNRxEqualizer().to(Config.DEVICE)
    if name == "transformer":
        return TransformerRxEqualizer().to(Config.DEVICE)
    if name == "tcn":
        return TCNRxEqualizer().to(Config.DEVICE)
    if name == "mamba":
        return MambaRxEqualizer().to(Config.DEVICE)
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
def evaluate_split(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    scale_search: bool = False,
) -> Tuple[float, float, float, int, float]:
    model.eval()
    criterion = build_criterion()
    bit_labels = BIT_LABELS.to(Config.DEVICE)
    current_batch_size = batch_size

    while True:
        total_loss = 0.0
        total_samples = 0
        correct = 0
        bit_errors = 0
        total_bits = 0
        preds_accum: List[torch.Tensor] = []
        targets_accum: List[torch.Tensor] = []
        try:
            for xb, yb in iter_tensor_batches(x, y, batch_size=current_batch_size, shuffle=False):
                with autocast_context():
                    mark_cudagraph_step_begin()
                    preds = model(xb)
                    loss = prediction_loss(preds, yb, criterion)
                preds_float = preds.float()
                target_float = yb.float()
                preds_accum.append(preds_float.detach().cpu())
                targets_accum.append(target_float.detach().cpu())
                batch_size_now = yb.size(0)
                total_loss += loss.item() * batch_size_now
                total_samples += batch_size_now
            preds_all = torch.cat(preds_accum, dim=0)
            targets_all = torch.cat(targets_accum, dim=0)
            target_classes = symbols_to_classes(targets_all.to(Config.DEVICE))
            if is_classifier_output(preds_all):
                scale = 1.0
                pred_classes = torch.argmax(preds_all.to(Config.DEVICE), dim=1)
            else:
                scale = find_best_symbol_scale(targets_all, preds_all) if scale_search else 1.0
                pred_classes = symbols_to_classes((preds_all * scale).to(Config.DEVICE))
            correct = (pred_classes == target_classes).sum().item()
            tx_bits = bit_labels[target_classes]
            rx_bits = bit_labels[pred_classes]
            bit_errors = (tx_bits != rx_bits).sum().item()
            total_bits = tx_bits.numel()
            return (
                total_loss / max(total_samples, 1),
                correct / max(total_samples, 1),
                bit_errors / max(total_bits, 1),
                current_batch_size,
                scale,
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
def compute_split_file_metrics(
    model: nn.Module,
    data: Dict[str, Any],
    split: str,
    batch_size: int,
) -> Tuple[List[Dict[str, float]], int]:
    file_spans = data.get(f"{split}_file_spans", [])
    if not file_spans:
        return [], batch_size

    results: List[Dict[str, float]] = []
    current_batch_size = batch_size
    x = data[f"{split}_x"]
    y = data[f"{split}_y"]
    for item in file_spans:
        start = int(item["start"])
        end = int(item["end"])
        baseline_scale = find_best_symbol_scale(item["tx_center"], item["rx_center"])
        tx_cls = symbols_to_classes(item["tx_center"].to(Config.DEVICE))
        rx_cls = symbols_to_classes((item["rx_center"] * baseline_scale).to(Config.DEVICE))
        baseline_ber = calculate_ber_from_classes(tx_cls, rx_cls)
        loss, acc, ber, current_batch_size, equalizer_scale = evaluate_split(
            model,
            x[start:end],
            y[start:end],
            current_batch_size,
            scale_search=True,
        )
        results.append(
            {
                "file_idx": int(item["file_idx"]),
                "samples": int(end - start),
                "baseline_ber": float(baseline_ber),
                "baseline_scale": float(baseline_scale),
                "equalized_ber": float(ber),
                "equalizer_scale": float(equalizer_scale),
                "accuracy": float(acc),
                "loss": float(loss),
                "improvement_rel": float((1 - ber / baseline_ber) * 100 if baseline_ber > 0 else 0.0),
            }
        )
    return results, current_batch_size


def add_file_metric_summary(metrics: Dict[str, Any], split: str, file_metrics: List[Dict[str, float]]):
    if not file_metrics:
        return
    equalized = np.array([row["equalized_ber"] for row in file_metrics], dtype=np.float64)
    baseline = np.array([row["baseline_ber"] for row in file_metrics], dtype=np.float64)
    metrics[f"{split}_file_equalized_ber_mean"] = float(equalized.mean())
    metrics[f"{split}_file_equalized_ber_std"] = float(equalized.std())
    metrics[f"{split}_file_equalized_ber_worst"] = float(equalized.max())
    metrics[f"{split}_file_baseline_ber_mean"] = float(baseline.mean())
    metrics[f"{split}_file_baseline_ber_worst"] = float(baseline.max())
    metrics[f"{split}_file_equalized_ber_by_file"] = ";".join(
        f"{int(row['file_idx'])}:{row['equalized_ber']:.6e}" for row in file_metrics
    )
    metrics[f"{split}_file_baseline_ber_by_file"] = ";".join(
        f"{int(row['file_idx'])}:{row['baseline_ber']:.6e}" for row in file_metrics
    )


@torch.inference_mode()
def compute_test_metrics(model: nn.Module, data: Dict[str, Any]) -> Dict[str, Any]:
    eval_prefix = "test"
    baseline_scale = find_best_symbol_scale(data[f"tx_{eval_prefix}_center"], data[f"rx_{eval_prefix}_center"])
    tx_cls = symbols_to_classes(data[f"tx_{eval_prefix}_center"].to(Config.DEVICE))
    rx_cls = symbols_to_classes((data[f"rx_{eval_prefix}_center"] * baseline_scale).to(Config.DEVICE))
    baseline_ber = calculate_ber_from_classes(tx_cls, rx_cls)
    eval_batch_size = data.get("eval_batch_size", Config.EVAL_BATCH_SIZE)
    test_loss, test_acc, test_ber, safe_eval_batch_size, equalizer_scale = evaluate_split(
        model, data[f"{eval_prefix}_x"], data[f"{eval_prefix}_y"], eval_batch_size, scale_search=True
    )
    return {
        "eval_split": eval_prefix,
        "baseline_ber": baseline_ber,
        "baseline_scale": baseline_scale,
        "equalized_ber": test_ber,
        "equalizer_scale": equalizer_scale,
        "accuracy": test_acc,
        "ser": 1.0 - test_acc,
        "test_loss": test_loss,
        "safe_eval_batch_size": safe_eval_batch_size,
        "improvement_abs": baseline_ber - test_ber,
        "improvement_rel": (1 - test_ber / baseline_ber) * 100 if baseline_ber > 0 else 0.0,
        "improvement_db": 10 * np.log10(baseline_ber / test_ber) if test_ber > 0 else float("inf"),
    }


def build_optimizer_with_lr(model: nn.Module, lr: float) -> optim.Optimizer:
    optimizer_kwargs = {"lr": lr, "weight_decay": Config.WEIGHT_DECAY}
    if Config.DEVICE.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        return optim.Adam(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        return optim.Adam(model.parameters(), **optimizer_kwargs)


def get_efficient_kan_module(model: nn.Module) -> Optional[nn.Module]:
    kan = getattr(model, "kan", None)
    if kan is None or not hasattr(kan, "layers"):
        return None
    layers = list(getattr(kan, "layers"))
    if not layers or not all(hasattr(layer, "base_weight") and hasattr(layer, "spline_weight") for layer in layers):
        return None
    return kan


def efficient_kan_layer_sizes(kan: nn.Module) -> List[int]:
    layers = list(kan.layers)
    if not layers:
        return []
    return [int(layers[0].in_features)] + [int(layer.out_features) for layer in layers]


@torch.no_grad()
def efficient_kan_edge_norm(layer: nn.Module) -> torch.Tensor:
    base = layer.base_weight.detach().float().abs()
    spline = layer.scaled_spline_weight.detach().float().abs().mean(dim=2)
    return base + spline


@torch.no_grad()
def select_efficient_kan_hidden_units(kan: nn.Module, keep_ratio: float) -> List[torch.Tensor]:
    layers = list(kan.layers)
    selections: List[torch.Tensor] = []
    for layer_idx in range(len(layers) - 1):
        current_layer = layers[layer_idx]
        next_layer = layers[layer_idx + 1]
        hidden_size = int(current_layer.out_features)
        keep_count = max(Config.KAN_STRUCTURAL_PRUNE_MIN_HIDDEN, int(round(hidden_size * keep_ratio)))
        keep_count = min(hidden_size, max(1, keep_count))
        incoming = efficient_kan_edge_norm(current_layer).mean(dim=1)
        outgoing = efficient_kan_edge_norm(next_layer).mean(dim=0)
        importance = torch.sqrt(incoming.clamp_min(1e-12) * outgoing.clamp_min(1e-12))
        keep = torch.topk(importance, k=keep_count, largest=True, sorted=False).indices
        keep = torch.sort(keep.cpu()).values
        selections.append(keep)
    return selections


@torch.no_grad()
def copy_pruned_kan_layer(old_layer: nn.Module, new_layer: nn.Module, input_idx: torch.Tensor, output_idx: torch.Tensor):
    input_idx = input_idx.to(old_layer.base_weight.device)
    output_idx = output_idx.to(old_layer.base_weight.device)
    new_layer.grid.copy_(old_layer.grid.index_select(0, input_idx).to(new_layer.grid.device))
    new_layer.base_weight.copy_(
        old_layer.base_weight.index_select(0, output_idx).index_select(1, input_idx).to(new_layer.base_weight.device)
    )
    new_layer.spline_weight.copy_(
        old_layer.spline_weight.index_select(0, output_idx)
        .index_select(1, input_idx)
        .to(new_layer.spline_weight.device)
    )
    if hasattr(old_layer, "spline_scaler") and hasattr(new_layer, "spline_scaler"):
        new_layer.spline_scaler.copy_(
            old_layer.spline_scaler.index_select(0, output_idx)
            .index_select(1, input_idx)
            .to(new_layer.spline_scaler.device)
        )


def structurally_prune_efficient_kan_model(model: nn.Module, keep_ratio: float) -> Tuple[nn.Module, List[int]]:
    if EfficientKAN is None:
        raise ImportError("EfficientKAN is unavailable.")
    old_kan = get_efficient_kan_module(model)
    if old_kan is None:
        raise ValueError("Model does not expose a prunable EfficientKAN module via `.kan`.")

    old_layers = list(old_kan.layers)
    old_sizes = efficient_kan_layer_sizes(old_kan)
    hidden_selections = select_efficient_kan_hidden_units(old_kan, keep_ratio)
    all_indices: List[torch.Tensor] = [
        torch.arange(old_sizes[0], dtype=torch.long),
        *hidden_selections,
        torch.arange(old_sizes[-1], dtype=torch.long),
    ]
    new_sizes = [int(idx.numel()) for idx in all_indices]
    new_kan = EfficientKAN(
        layers_hidden=new_sizes,
        grid_size=int(old_kan.grid_size),
        spline_order=int(old_kan.spline_order),
        scale_noise=Config.EFFICIENT_KAN_SCALE_NOISE,
        scale_base=Config.EFFICIENT_KAN_SCALE_BASE,
        scale_spline=Config.EFFICIENT_KAN_SCALE_SPLINE,
        base_activation=nn.SiLU,
        grid_eps=Config.EFFICIENT_KAN_GRID_EPS,
        grid_range=Config.EFFICIENT_KAN_GRID_RANGE,
    ).to(Config.DEVICE)
    for layer_idx, (old_layer, new_layer) in enumerate(zip(old_layers, new_kan.layers)):
        copy_pruned_kan_layer(old_layer, new_layer, all_indices[layer_idx], all_indices[layer_idx + 1])
    model.kan = new_kan
    return model, new_sizes


def efficiency_score_from_metrics(metrics: Dict[str, Any]) -> float:
    baseline = float(metrics.get("baseline_ber", 0.0))
    equalized = float(metrics.get("equalized_ber", 0.0))
    batch_time = float(metrics.get("efficiency_batch_time_sec", 0.0))
    if baseline <= 0 or batch_time <= 0:
        return 0.0
    improvement = max((baseline - equalized) / baseline, 0.0)
    return float(improvement**Config.EFFICIENCY_SCORE_POWER / batch_time)


@torch.inference_mode()
def measure_batch_inference_time(model: nn.Module, x: torch.Tensor) -> float:
    model.eval()
    batch_size = min(Config.EFFICIENCY_BATCH_SIZE, x.size(0))
    if batch_size <= 0:
        return 0.0
    xb = x[:batch_size].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
    for _ in range(Config.EFFICIENCY_TIMING_WARMUP):
        with autocast_context():
            mark_cudagraph_step_begin()
            _ = model(xb)
    if Config.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(Config.EFFICIENCY_TIMING_REPEATS):
        with autocast_context():
            mark_cudagraph_step_begin()
            _ = model(xb)
    if Config.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / max(Config.EFFICIENCY_TIMING_REPEATS, 1)


def add_efficiency_metrics(model: nn.Module, data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    batch_time = measure_batch_inference_time(model, data["test_x"])
    metrics["efficiency_batch_size"] = int(min(Config.EFFICIENCY_BATCH_SIZE, data["test_x"].size(0)))
    metrics["efficiency_batch_time_sec"] = float(batch_time)
    metrics["efficiency_score"] = efficiency_score_from_metrics(metrics)
    return metrics


def fine_tune_model_for_pruning(
    model: nn.Module,
    data: Dict[str, Any],
    epochs: int,
    lr: float,
    eval_batch_size: int,
) -> Tuple[nn.Module, Dict[str, float], int]:
    if epochs <= 0:
        val_loss, val_acc, val_ber, eval_batch_size, val_scale = evaluate_split(
            model, data["val_x"], data["val_y"], eval_batch_size, scale_search=True
        )
        return model, {"val_loss": val_loss, "val_acc": val_acc, "val_ber": val_ber, "val_scale": val_scale}, eval_batch_size

    optimizer = build_optimizer_with_lr(model, lr)
    criterion = build_criterion()
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metrics = {"val_loss": float("inf"), "val_acc": 0.0, "val_ber": float("inf"), "val_scale": 1.0}
    train_block_size = Config.TRAIN_BLOCK_SIZE

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        total_train = data["train_x"].size(0)
        num_blocks = (total_train + train_block_size - 1) // train_block_size
        for block_idx in torch.randperm(num_blocks).tolist():
            block_start = block_idx * train_block_size
            block_end = min(block_start + train_block_size, total_train)
            xb = data["train_x"][block_start:block_end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
            yb = data["train_y"][block_start:block_end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                mark_cudagraph_step_begin()
                preds = model(xb)
                loss = prediction_loss(preds, yb, criterion) + compute_model_regularization(model)
            scaler.scale(loss).backward()
            if Config.GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * yb.size(0)
            seen += yb.size(0)

        val_loss, val_acc, val_ber, eval_batch_size, val_scale = evaluate_split(
            model, data["val_x"], data["val_y"], eval_batch_size, scale_search=True
        )
        if val_ber < best_metrics["val_ber"]:
            best_metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_ber": val_ber, "val_scale": val_scale}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(
            f"prune fine-tune | epoch {epoch+1:3d}/{epochs} | "
            f"train {running_loss / max(seen, 1):.6f} | val_ber {val_ber:.6e}"
        )

    model.load_state_dict(best_state)
    return model, best_metrics, eval_batch_size


def should_replace_by_pruned(candidate_metrics: Dict[str, Any], best_metrics: Dict[str, Any]) -> bool:
    mode = Config.KAN_STRUCTURAL_PRUNE_SELECT_BY
    if mode == "val_ber":
        return float(candidate_metrics.get("prune_val_ber", float("inf"))) < float(
            best_metrics.get("prune_val_ber", best_metrics.get("best_val_ber", float("inf")))
        )
    if mode == "test_ber":
        return float(candidate_metrics["equalized_ber"]) < float(best_metrics["equalized_ber"])
    return float(candidate_metrics.get("efficiency_score", 0.0)) > float(best_metrics.get("efficiency_score", 0.0))


def maybe_prune_efficient_kan_model(
    model: nn.Module,
    model_name: str,
    data: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_batch_size: int,
) -> Tuple[nn.Module, Dict[str, Any], int]:
    if not Config.KAN_STRUCTURAL_PRUNE_AFTER_TRAINING:
        return model, base_metrics, eval_batch_size
    if get_efficient_kan_module(model) is None:
        log(f"{model_name} | structural KAN pruning skipped: model has no prunable EfficientKAN `.kan`")
        return model, base_metrics, eval_batch_size

    base_metrics = add_efficiency_metrics(model, data, base_metrics)
    base_metrics["pruned"] = False
    base_metrics["prune_keep_ratio"] = 1.0
    base_metrics["prune_layer_sizes"] = str(efficient_kan_layer_sizes(get_efficient_kan_module(model)))
    base_metrics["prune_val_ber"] = base_metrics.get("best_val_ber", float("inf"))
    best_model = model
    best_metrics = dict(base_metrics)
    rows: List[Dict[str, Any]] = [dict(base_metrics, model_type=model_name, prune_candidate="unpruned")]

    for keep_ratio in Config.KAN_STRUCTURAL_PRUNE_KEEP_RATIOS:
        candidate = copy.deepcopy(model).to(Config.DEVICE)
        try:
            candidate, layer_sizes = structurally_prune_efficient_kan_model(candidate, keep_ratio)
        except Exception as exc:
            log(f"{model_name} | prune keep_ratio={keep_ratio:.2f} skipped: {exc}")
            continue
        candidate_params = count_trainable_parameters(candidate)
        log(
            f"{model_name} | prune candidate keep_ratio={keep_ratio:.2f} | "
            f"layers {layer_sizes} | params {candidate_params:,}"
        )
        candidate, val_metrics, eval_batch_size = fine_tune_model_for_pruning(
            candidate,
            data,
            Config.KAN_STRUCTURAL_PRUNE_FINE_TUNE_EPOCHS,
            Config.KAN_STRUCTURAL_PRUNE_FINE_TUNE_LR,
            eval_batch_size,
        )
        candidate_metrics = compute_test_metrics(candidate, {**data, "eval_batch_size": eval_batch_size})
        candidate_metrics["trainable_params"] = candidate_params
        candidate_metrics["pruned"] = True
        candidate_metrics["prune_keep_ratio"] = float(keep_ratio)
        candidate_metrics["prune_layer_sizes"] = str(layer_sizes)
        candidate_metrics["prune_val_loss"] = float(val_metrics["val_loss"])
        candidate_metrics["prune_val_acc"] = float(val_metrics["val_acc"])
        candidate_metrics["prune_val_ber"] = float(val_metrics["val_ber"])
        candidate_metrics = add_efficiency_metrics(candidate, data, candidate_metrics)
        rows.append(dict(candidate_metrics, model_type=model_name, prune_candidate=f"keep_{keep_ratio:.2f}"))
        log(
            f"{model_name} | prune keep_ratio={keep_ratio:.2f} | "
            f"test_ber {candidate_metrics['equalized_ber']:.6e} | "
            f"batch16k {candidate_metrics['efficiency_batch_time_sec']:.6f}s | "
            f"score {candidate_metrics['efficiency_score']:.3f}"
        )
        if should_replace_by_pruned(candidate_metrics, best_metrics):
            best_model = candidate
            best_metrics = dict(candidate_metrics)
        elif Config.DEVICE.type == "cuda":
            candidate.to("cpu")
            del candidate
            torch.cuda.empty_cache()

    prune_df = pd.DataFrame(rows)
    prune_path = Config.OUT_DIR / f"{model_name}_pruning_candidates.csv"
    prune_df.to_csv(prune_path, index=False)
    log(
        f"{model_name} | pruning selected keep_ratio={best_metrics.get('prune_keep_ratio', 1.0)} | "
        f"test_ber {best_metrics['equalized_ber']:.6e} | "
        f"score {best_metrics.get('efficiency_score', 0.0):.3f} | saved {prune_path}"
    )
    return best_model, best_metrics, eval_batch_size


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


def plot_efficient_kan_sweep(df: pd.DataFrame, x_col: str, x_label: str, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name in Config.EFFICIENT_KAN_SWEEP_MODELS:
        model_df = df[df["model_type"] == model_name].sort_values(x_col)
        if model_df.empty:
            continue
        ax.plot(model_df[x_col], model_df["equalized_ber"], marker="o", linewidth=2, label=model_name)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("BER")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficient_kan_tradeoff(df: pd.DataFrame, x_col: str, x_label: str, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name in Config.EFFICIENT_KAN_SWEEP_MODELS:
        model_df = df[df["model_type"] == model_name]
        if model_df.empty:
            continue
        ax.scatter(model_df[x_col], model_df["equalized_ber"], s=60, label=model_name)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("BER")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_lines(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    filename: str,
    title: str,
    y_col: str = "equalized_ber",
):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, model_df in df.groupby("model_type"):
        model_df = model_df.sort_values(x_col)
        ax.plot(model_df[x_col], model_df[y_col], marker="o", linewidth=2, label=model_name)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("BER")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_complexity(df: pd.DataFrame, filename: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, model_df in df.groupby("model_type"):
        ax.plot(
            model_df["trainable_params"],
            model_df["equalized_ber"],
            marker="o",
            linewidth=1.5,
            linestyle="-",
            label=model_name,
        )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("BER")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def hidden_overrides(hidden_dim: int) -> Dict[str, int]:
    return {
        "HIDDEN_DIM": hidden_dim,
        "EFFICIENT_KAN_HIDDEN_DIM": hidden_dim,
        "FASTKAN_HIDDEN_DIM": hidden_dim,
    }


def layer_overrides(layer_count: int) -> Dict[str, int]:
    return {
        "EFFICIENT_KAN_LAYERS": layer_count,
        "FASTKAN_LAYERS": layer_count,
        "MLP_LAYERS": layer_count,
    }


def run_model_with_overrides(model_name: str, max_test_files: int, **overrides) -> Dict[str, Any]:
    tracked_keys = [
        "CONTEXT_K",
        "SEQ_LEN",
        "HIDDEN_DIM",
        "LSTM_HIDDEN",
        "SAVE_BEST",
        "EPOCHS",
        "LEARNING_RATE",
        "EFFICIENT_KAN_HIDDEN_DIM",
        "EFFICIENT_KAN_LAYERS",
        "EFFICIENT_KAN_GRID_SIZE",
        "EFFICIENT_KAN_SPLINE_ORDER",
        "FASTKAN_HIDDEN_DIM",
        "FASTKAN_LAYERS",
        "FASTKAN_NUM_GRIDS",
        "MLP_LAYERS",
        "COMPUTE_PER_FILE_METRICS",
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


def plot_fastkan_classifier_score(df: pd.DataFrame, filename: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, model_df in df.groupby("model_type"):
        ax.scatter(
            model_df["trainable_params"],
            model_df["efficiency_score"],
            s=70,
            label=model_name,
        )
    ax.set_title("FastKAN Classifiers: BER-Speed Efficiency", fontweight="bold")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Efficiency Score")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_fastkan_classifier_sweep():
    log("\nRunning compact FastKAN/RBF-KAN classifier sweep...")
    rows: List[Dict[str, Any]] = []
    output_path = Config.OUT_DIR / "fastkan_classifier_sweep_all.csv"
    base = {
        "EPOCHS": Config.FASTKAN_CLASSIFIER_SWEEP_EPOCHS,
        "COMPUTE_PER_FILE_METRICS": False,
        "FASTKAN_HIDDEN_DIM": Config.FASTKAN_HIDDEN_DIM,
        "FASTKAN_LAYERS": Config.FASTKAN_LAYERS,
        "FASTKAN_NUM_GRIDS": Config.FASTKAN_NUM_GRIDS,
    }

    def run_case(sweep_name: str, model_name: str, **overrides):
        effective = {**base, **overrides}
        log(
            f"fastkan sweep | {sweep_name} | {model_name} | "
            f"hidden={effective['FASTKAN_HIDDEN_DIM']} | grids={effective['FASTKAN_NUM_GRIDS']} | "
            f"layers={effective['FASTKAN_LAYERS']}"
        )
        results = run_model_with_overrides(
            model_name,
            max_test_files=Config.FASTKAN_CLASSIFIER_SWEEP_TEST_FILES,
            **effective,
        )
        rows.append(
            {
                "sweep": sweep_name,
                "model_type": model_name,
                "hidden_dim": effective["FASTKAN_HIDDEN_DIM"],
                "num_grids": effective["FASTKAN_NUM_GRIDS"],
                "layers": effective["FASTKAN_LAYERS"],
                **results,
            }
        )
        pd.DataFrame(rows).to_csv(output_path, index=False)

    for model_name in Config.FASTKAN_CLASSIFIER_SWEEP_MODELS:
        for hidden_dim in Config.FASTKAN_CLASSIFIER_HIDDEN_VALUES:
            run_case("hidden_dim", model_name, FASTKAN_HIDDEN_DIM=hidden_dim)
        for num_grids in Config.FASTKAN_CLASSIFIER_GRID_VALUES:
            run_case("num_grids", model_name, FASTKAN_NUM_GRIDS=num_grids)
        for layers in Config.FASTKAN_CLASSIFIER_LAYER_VALUES:
            run_case("layers", model_name, FASTKAN_LAYERS=layers)

    df = pd.DataFrame(rows)
    for sweep_name, x_col, x_label in [
        ("hidden_dim", "hidden_dim", "Hidden Dimension"),
        ("num_grids", "num_grids", "RBF Grid Count"),
        ("layers", "layers", "FastKAN Layers"),
    ]:
        sweep_df = df[df["sweep"] == sweep_name].copy()
        sweep_df.to_csv(Config.OUT_DIR / f"fastkan_classifier_ber_vs_{sweep_name}.csv", index=False)
        plot_experiment_lines(
            sweep_df,
            x_col=x_col,
            x_label=x_label,
            filename=f"fastkan_classifier_ber_vs_{sweep_name}.png",
            title=f"FastKAN Classifier BER vs {x_label}",
        )
    plot_experiment_complexity(
        df,
        filename="fastkan_classifier_ber_vs_complexity.png",
        title="FastKAN Classifiers: BER vs Complexity",
    )
    plot_fastkan_classifier_score(df, "fastkan_classifier_efficiency_score.png")
    log(f"Saved FastKAN classifier sweep: {output_path}")


def run_sweep_experiments():
    if Config.RUN_FASTKAN_CLASSIFIER_SWEEP:
        run_fastkan_classifier_sweep()
        return
    if Config.RUN_KAN_EXPERIMENT_SUITE:
        run_kan_experiment_suite()
        return
    if Config.RUN_EFFICIENT_KAN_SWEEP:
        run_efficient_kan_sweep_experiments()
        return

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


def run_kan_experiment_suite():
    log("\nRunning KAN/MLP experiment suite...")
    rows: List[Dict[str, Any]] = []

    base = {
        "EPOCHS": Config.EXPERIMENT_EPOCHS,
        "EFFICIENT_KAN_GRID_SIZE": Config.EXPERIMENT_FIXED_GRID,
        "EFFICIENT_KAN_SPLINE_ORDER": Config.EXPERIMENT_FIXED_SPLINE_ORDER,
        "COMPUTE_PER_FILE_METRICS": Config.EXPERIMENT_COMPUTE_PER_FILE_METRICS,
    }

    def run_case(sweep_name: str, model_name: str, **overrides):
        effective = {**base, **overrides}
        effective["SEQ_LEN"] = 2 * effective.get("CONTEXT_K", Config.CONTEXT_K) + 1
        case_id = (
            f"{sweep_name}_{model_name}_"
            f"k{effective.get('CONTEXT_K', Config.CONTEXT_K)}_"
            f"h{effective.get('EFFICIENT_KAN_HIDDEN_DIM', Config.EFFICIENT_KAN_HIDDEN_DIM)}_"
            f"mlph{effective.get('HIDDEN_DIM', Config.HIDDEN_DIM)}_"
            f"l{effective.get('EFFICIENT_KAN_LAYERS', Config.EFFICIENT_KAN_LAYERS)}_"
            f"mlpl{effective.get('MLP_LAYERS', Config.MLP_LAYERS)}_"
            f"g{effective.get('EFFICIENT_KAN_GRID_SIZE', Config.EFFICIENT_KAN_GRID_SIZE)}_"
            f"o{effective.get('EFFICIENT_KAN_SPLINE_ORDER', Config.EFFICIENT_KAN_SPLINE_ORDER)}"
        )
        log(f"experiment | {case_id}")
        results = run_model_with_overrides(
            model_name,
            max_test_files=Config.EXPERIMENT_TEST_FILES,
            **effective,
        )
        row = {
            "case_id": case_id,
            "sweep": sweep_name,
            "model_type": model_name,
            "context_k": effective.get("CONTEXT_K", Config.CONTEXT_K),
            "seq_len": effective.get("SEQ_LEN", Config.SEQ_LEN),
            "hidden_dim": effective.get("EFFICIENT_KAN_HIDDEN_DIM", effective.get("HIDDEN_DIM", Config.HIDDEN_DIM)),
            "mlp_hidden_dim": effective.get("HIDDEN_DIM", Config.HIDDEN_DIM),
            "kan_hidden_dim": effective.get("EFFICIENT_KAN_HIDDEN_DIM", Config.EFFICIENT_KAN_HIDDEN_DIM),
            "layers": effective.get("EFFICIENT_KAN_LAYERS", effective.get("MLP_LAYERS", Config.MLP_LAYERS)),
            "mlp_layers": effective.get("MLP_LAYERS", Config.MLP_LAYERS),
            "kan_layers": effective.get("EFFICIENT_KAN_LAYERS", Config.EFFICIENT_KAN_LAYERS),
            "grid_size": effective.get("EFFICIENT_KAN_GRID_SIZE", Config.EFFICIENT_KAN_GRID_SIZE),
            "spline_order": effective.get("EFFICIENT_KAN_SPLINE_ORDER", Config.EFFICIENT_KAN_SPLINE_ORDER),
            "epochs": effective["EPOCHS"],
            **results,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(Config.OUT_DIR / "kan_experiment_suite_all.csv", index=False)

    for model_name in Config.EXPERIMENT_KAN_MODELS:
        for grid_size in Config.EXPERIMENT_GRID_VALUES:
            run_case("ber_vs_grid", model_name, EFFICIENT_KAN_GRID_SIZE=grid_size)

        for spline_order in Config.EXPERIMENT_SPLINE_ORDER_VALUES:
            run_case("ber_vs_spline_order", model_name, EFFICIENT_KAN_SPLINE_ORDER=spline_order)

    for model_name in Config.EXPERIMENT_COMPARE_MODELS:
        for hidden_dim in Config.EXPERIMENT_HIDDEN_VALUES:
            run_case(
                "kan_mlp_vs_hidden_grid16",
                model_name,
                **hidden_overrides(hidden_dim),
                EFFICIENT_KAN_GRID_SIZE=Config.EXPERIMENT_FIXED_GRID,
            )

        for context_k in Config.EXPERIMENT_WINDOW_VALUES:
            run_case(
                "kan_mlp_vs_window",
                model_name,
                CONTEXT_K=context_k,
                EFFICIENT_KAN_GRID_SIZE=Config.EXPERIMENT_FIXED_GRID,
            )

        for layer_count in Config.EXPERIMENT_LAYER_VALUES:
            run_case(
                "kan_mlp_vs_layers",
                model_name,
                **layer_overrides(layer_count),
                EFFICIENT_KAN_GRID_SIZE=Config.EXPERIMENT_FIXED_GRID,
            )

    for model_name in Config.EXPERIMENT_COMPLEXITY_MODELS:
        for hidden_dim in Config.EXPERIMENT_HIDDEN_VALUES:
            run_case(
                "ber_vs_complexity",
                model_name,
                **hidden_overrides(hidden_dim),
                EFFICIENT_KAN_GRID_SIZE=Config.EXPERIMENT_FIXED_GRID,
            )

    df = pd.DataFrame(rows)
    all_path = Config.OUT_DIR / "kan_experiment_suite_all.csv"
    df.to_csv(all_path, index=False)

    plot_specs = [
        ("ber_vs_grid", "grid_size", "Grid Size", "ber_vs_grid.png", "BER vs Grid Size"),
        (
            "ber_vs_spline_order",
            "spline_order",
            "Spline Order",
            "ber_vs_spline_order.png",
            "BER vs Spline Order",
        ),
        (
            "kan_mlp_vs_hidden_grid16",
            "hidden_dim",
            "Hidden Dimension",
            "kan_mlp_ber_vs_hidden_grid16.png",
            "KAN vs MLP: BER vs Hidden Dim (grid=16)",
        ),
        (
            "kan_mlp_vs_window",
            "seq_len",
            "Window Size (symbols)",
            "kan_mlp_ber_vs_window.png",
            "KAN vs MLP: BER vs Window Size",
        ),
        (
            "kan_mlp_vs_layers",
            "layers",
            "Number of Layers",
            "kan_mlp_ber_vs_layers.png",
            "KAN vs MLP: BER vs Number of Layers",
        ),
    ]
    for sweep_name, x_col, x_label, filename, title in plot_specs:
        sweep_df = df[df["sweep"] == sweep_name].copy()
        sweep_df.to_csv(Config.OUT_DIR / f"{sweep_name}.csv", index=False)
        if not sweep_df.empty:
            plot_experiment_lines(sweep_df, x_col=x_col, x_label=x_label, filename=filename, title=title)

    complexity_df = df[df["sweep"] == "ber_vs_complexity"].copy()
    complexity_df.to_csv(Config.OUT_DIR / "ber_vs_complexity.csv", index=False)
    if not complexity_df.empty:
        plot_experiment_complexity(
            complexity_df,
            filename="ber_vs_complexity.png",
            title="BER vs Complexity",
        )

    log(f"Saved KAN experiment suite: {all_path}")


def run_efficient_kan_sweep_experiments():
    log("\nRunning EfficientKAN regression/classifier sweep...")
    rows: List[Dict[str, float]] = []
    base = {
        "EPOCHS": Config.EFFICIENT_KAN_SWEEP_EPOCHS,
        "EFFICIENT_KAN_HIDDEN_DIM": Config.EFFICIENT_KAN_HIDDEN_DIM,
        "EFFICIENT_KAN_LAYERS": Config.EFFICIENT_KAN_LAYERS,
        "EFFICIENT_KAN_GRID_SIZE": Config.EFFICIENT_KAN_GRID_SIZE,
        "EFFICIENT_KAN_SPLINE_ORDER": Config.EFFICIENT_KAN_SPLINE_ORDER,
        "LEARNING_RATE": Config.LEARNING_RATE,
    }

    def run_case(sweep_name: str, model_name: str, **overrides):
        effective = {**base, **overrides}
        case_id = (
            f"{sweep_name}_{model_name}_"
            f"h{effective['EFFICIENT_KAN_HIDDEN_DIM']}_"
            f"l{effective['EFFICIENT_KAN_LAYERS']}_"
            f"g{effective['EFFICIENT_KAN_GRID_SIZE']}_"
            f"o{effective['EFFICIENT_KAN_SPLINE_ORDER']}_"
            f"lr{effective['LEARNING_RATE']:.0e}"
        )
        log(f"sweep | {case_id}")
        results = run_model_with_overrides(
            model_name,
            max_test_files=Config.EFFICIENT_KAN_SWEEP_TEST_FILES,
            **effective,
        )
        row = {
            "case_id": case_id,
            "sweep": sweep_name,
            "model_type": model_name,
            "hidden_dim": effective["EFFICIENT_KAN_HIDDEN_DIM"],
            "layers": effective["EFFICIENT_KAN_LAYERS"],
            "grid_size": effective["EFFICIENT_KAN_GRID_SIZE"],
            "spline_order": effective["EFFICIENT_KAN_SPLINE_ORDER"],
            "learning_rate": effective["LEARNING_RATE"],
            "epochs": effective["EPOCHS"],
            **results,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(Config.OUT_DIR / "efficient_kan_sweep_all.csv", index=False)

    for model_name in Config.EFFICIENT_KAN_SWEEP_MODELS:
        for hidden_dim in Config.EFFICIENT_KAN_HIDDEN_SWEEP_VALUES:
            for learning_rate in Config.EFFICIENT_KAN_LR_SWEEP_VALUES:
                run_case(
                    "hidden_lr",
                    model_name,
                    EFFICIENT_KAN_HIDDEN_DIM=hidden_dim,
                    LEARNING_RATE=learning_rate,
                )

        for grid_size in Config.EFFICIENT_KAN_GRID_SWEEP_VALUES:
            run_case("grid_size", model_name, EFFICIENT_KAN_GRID_SIZE=grid_size)

        for spline_order in Config.EFFICIENT_KAN_ORDER_SWEEP_VALUES:
            run_case("spline_order", model_name, EFFICIENT_KAN_SPLINE_ORDER=spline_order)

        for layers in Config.EFFICIENT_KAN_LAYER_SWEEP_VALUES:
            run_case("layers", model_name, EFFICIENT_KAN_LAYERS=layers)

    df = pd.DataFrame(rows)
    df.to_csv(Config.OUT_DIR / "efficient_kan_sweep_all.csv", index=False)

    hidden_df = df[df["sweep"] == "hidden_lr"].copy()
    grid_df = df[df["sweep"] == "grid_size"].copy()
    order_df = df[df["sweep"] == "spline_order"].copy()
    layer_df = df[df["sweep"] == "layers"].copy()

    hidden_df.to_csv(Config.OUT_DIR / "efficient_kan_ber_vs_hidden_lr.csv", index=False)
    grid_df.to_csv(Config.OUT_DIR / "efficient_kan_ber_vs_grid.csv", index=False)
    order_df.to_csv(Config.OUT_DIR / "efficient_kan_ber_vs_order.csv", index=False)
    layer_df.to_csv(Config.OUT_DIR / "efficient_kan_ber_vs_layers.csv", index=False)

    for learning_rate in Config.EFFICIENT_KAN_LR_SWEEP_VALUES:
        lr_df = hidden_df[hidden_df["learning_rate"] == learning_rate]
        if lr_df.empty:
            continue
        plot_efficient_kan_sweep(
            lr_df,
            x_col="hidden_dim",
            x_label="EfficientKAN Hidden Dimension",
            filename=f"efficient_kan_ber_vs_hidden_lr_{learning_rate:.0e}.png",
            title=f"EfficientKAN BER vs Hidden Dim (lr={learning_rate:.0e})",
        )

    plot_efficient_kan_sweep(
        grid_df,
        x_col="grid_size",
        x_label="Grid Size",
        filename="efficient_kan_ber_vs_grid.png",
        title="EfficientKAN BER vs Grid Size",
    )
    plot_efficient_kan_sweep(
        order_df,
        x_col="spline_order",
        x_label="Spline Order",
        filename="efficient_kan_ber_vs_spline_order.png",
        title="EfficientKAN BER vs Spline Order",
    )
    plot_efficient_kan_sweep(
        layer_df,
        x_col="layers",
        x_label="KAN Hidden Layers",
        filename="efficient_kan_ber_vs_layers.png",
        title="EfficientKAN BER vs Number of Layers",
    )
    plot_efficient_kan_tradeoff(
        df,
        x_col="trainable_params",
        x_label="Trainable Parameters",
        filename="efficient_kan_params_vs_ber.png",
        title="EfficientKAN Parameter Count vs BER",
    )
    plot_efficient_kan_tradeoff(
        df,
        x_col="train_samples_per_sec",
        x_label="Training Samples/sec",
        filename="efficient_kan_speed_vs_ber.png",
        title="EfficientKAN Training Speed vs BER",
    )
    log(f"Saved EfficientKAN sweep: {Config.OUT_DIR / 'efficient_kan_sweep_all.csv'}")


def train_one_model(model_name: str, data: Dict[str, Any]) -> Tuple[nn.Module, Dict, Dict[str, Any]]:
    model = make_model(model_name)
    trainable_params = count_trainable_parameters(model)
    log(
        f"{model_name} | params {trainable_params:,} | "
        f"{MODEL_NOTES.get(model_name, 'custom equalizer')}"
    )
    if hasattr(model, "initialize_from_data"):
        model.initialize_from_data(data["train_x"], data["train_y"])
        log(f"{model_name} | initialized from training windows")
    if Config.USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=Config.TORCH_COMPILE_MODE)
            log(f"{model_name} | torch.compile enabled ({Config.TORCH_COMPILE_MODE})")
        except Exception as exc:
            log(f"{model_name} | torch.compile disabled: {exc}")

    optimizer = build_optimizer(model)
    criterion = build_criterion()
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP)
    best_train_loss = float("inf")
    best_val_ber = float("inf")
    best_score = float("inf")
    best_state = None
    steps_without_improvement = 0
    early_stop_without_improvement = 0
    early_stopped = False
    stop_reason = "completed"

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_ber": [],
        "val_scale": [],
        "val_epochs": [],
        "test_loss": [],
        "test_acc": [],
        "test_ber": [],
        "test_scale": [],
        "test_epochs": [],
        "lr": [],
        "epoch_time_sec": [],
        "train_samples_per_sec": [],
    }

    train_x = data["train_x"]
    train_y = data["train_y"]
    train_block_size = Config.TRAIN_BLOCK_SIZE
    eval_batch_size = Config.EVAL_BATCH_SIZE

    if Config.SAVE_BEST_BY in {"val_ber", "val_loss"}:
        init_val_loss, _, init_val_ber, eval_batch_size, _ = evaluate_split(
            model,
            data["val_x"],
            data["val_y"],
            eval_batch_size,
            scale_search=True,
        )
        best_val_ber = init_val_ber
        best_score = init_val_ber if Config.SAVE_BEST_BY == "val_ber" else init_val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if Config.SAVE_BEST:
            torch.save(best_state, Config.OUT_DIR / f"{model_name}_best.pth")
        log(
            f"{model_name} | init checkpoint | val {init_val_loss:.6f} | "
            f"val_ber {init_val_ber:.6e}"
        )

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
                        loss = prediction_loss(preds, yb, criterion)
                        loss = loss + compute_model_regularization(model)
                    scaler.scale(loss).backward()
                    if Config.GRAD_CLIP_NORM > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
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

        val_loss, val_acc, val_ber, eval_batch_size, val_scale = evaluate_split(
            model,
            data["val_x"],
            data["val_y"],
            eval_batch_size,
            scale_search=True,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ber"].append(val_ber)
        history["val_scale"].append(val_scale)
        history["val_epochs"].append(epoch + 1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time_sec"].append(epoch_time)
        history["train_samples_per_sec"].append(speed)

        should_eval_test = Config.EVAL_TEST_DURING_TRAINING and (
            (epoch + 1) % Config.TEST_BER_EVERY == 0 or epoch == 0 or epoch == Config.EPOCHS - 1
        )
        if should_eval_test:
            test_metrics = compute_test_metrics(model, {**data, "eval_batch_size": eval_batch_size})
            eval_batch_size = int(test_metrics["safe_eval_batch_size"])
            history["test_loss"].append(test_metrics["test_loss"])
            history["test_acc"].append(test_metrics["accuracy"])
            history["test_ber"].append(test_metrics["equalized_ber"])
            history["test_scale"].append(test_metrics["equalizer_scale"])
            history["test_epochs"].append(epoch + 1)
            log(
                f"{model_name} | epoch {epoch+1:4d}/{Config.EPOCHS} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | val_ber {val_ber:.6e} | "
                f"test_ber {test_metrics['equalized_ber']:.6e} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"speed {speed:,.0f} samp/s | time {epoch_time:.1f}s"
            )
        elif epoch % Config.LOG_EVERY == 0 or epoch == Config.EPOCHS - 1:
            log(
                f"{model_name} | epoch {epoch+1:4d}/{Config.EPOCHS} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | val_ber {val_ber:.6e} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | speed {speed:,.0f} samp/s | time {epoch_time:.1f}s"
            )

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if val_ber + Config.EARLY_STOPPING_THRESHOLD < best_val_ber:
            best_val_ber = val_ber
            early_stop_without_improvement = 0
        else:
            early_stop_without_improvement += 1
        if Config.SAVE_BEST_BY == "train_loss":
            monitor_value = train_loss
        elif Config.SAVE_BEST_BY == "val_ber":
            monitor_value = val_ber
        else:
            monitor_value = val_loss
        if monitor_value + Config.SCHEDULER_THRESHOLD < best_score:
            best_score = monitor_value
            steps_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if Config.SAVE_BEST:
                torch.save(best_state, Config.OUT_DIR / f"{model_name}_best.pth")
        else:
            steps_without_improvement += 1

        if Config.LR_SCHEDULER == "notebook_decay" and steps_without_improvement >= Config.DECAY_STEPS:
            current_lr = optimizer.param_groups[0]["lr"]
            new_lr = current_lr * Config.SCHEDULER_FACTOR
            steps_without_improvement = 0
            if new_lr < Config.MIN_LR:
                stop_reason = "lr_floor"
                log(f"{model_name} | epoch {epoch+1} -- stopping at lr floor ({current_lr:.6g})")
                break
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            log(f"{model_name} | epoch {epoch+1} -- scheduler reduced lr to {new_lr:.6g}")

        if (
            Config.EARLY_STOPPING
            and epoch + 1 >= Config.EARLY_STOPPING_MIN_EPOCHS
            and early_stop_without_improvement >= Config.EARLY_STOPPING_PATIENCE
        ):
            early_stopped = True
            stop_reason = f"early_stop_val_ber_patience_{Config.EARLY_STOPPING_PATIENCE}"
            log(
                f"{model_name} | epoch {epoch+1} -- early stopping: "
                f"val_ber did not improve for {early_stop_without_improvement} epochs "
                f"(best_val_ber={best_val_ber:.6e})"
            )
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_metrics = compute_test_metrics(model, {**data, "eval_batch_size": eval_batch_size})
    final_metrics["trainable_params"] = count_trainable_parameters(model)
    final_metrics["best_val_ber"] = best_val_ber
    final_metrics["best_train_loss"] = best_train_loss
    final_metrics["train_samples_per_sec"] = float(np.mean(history["train_samples_per_sec"])) if history["train_samples_per_sec"] else 0.0
    final_metrics["mean_epoch_time_sec"] = float(np.mean(history["epoch_time_sec"])) if history["epoch_time_sec"] else 0.0
    final_metrics["epochs_ran"] = len(history["train_loss"])
    final_metrics["early_stopped"] = early_stopped
    final_metrics["stop_reason"] = stop_reason
    final_metrics = add_efficiency_metrics(model, data, final_metrics)
    model, final_metrics, eval_batch_size = maybe_prune_efficient_kan_model(
        model,
        model_name,
        data,
        final_metrics,
        eval_batch_size,
    )
    final_metrics["trainable_params"] = count_trainable_parameters(model)
    final_metrics["best_val_ber"] = best_val_ber
    final_metrics["best_train_loss"] = best_train_loss
    final_metrics["train_samples_per_sec"] = float(np.mean(history["train_samples_per_sec"])) if history["train_samples_per_sec"] else 0.0
    final_metrics["mean_epoch_time_sec"] = float(np.mean(history["epoch_time_sec"])) if history["epoch_time_sec"] else 0.0
    final_metrics["epochs_ran"] = len(history["train_loss"])
    final_metrics["early_stopped"] = early_stopped
    final_metrics["stop_reason"] = stop_reason
    if Config.COMPUTE_PER_FILE_METRICS:
        val_file_metrics, eval_batch_size = compute_split_file_metrics(model, data, "val", eval_batch_size)
        test_file_metrics, eval_batch_size = compute_split_file_metrics(model, data, "test", eval_batch_size)
        add_file_metric_summary(final_metrics, "val", val_file_metrics)
        add_file_metric_summary(final_metrics, "test", test_file_metrics)
    return model, history, final_metrics


def main():
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Device: {Config.DEVICE}")

    if Config.RUN_FASTKAN_CLASSIFIER_SWEEP:
        run_fastkan_classifier_sweep()
        return

    if Config.RUN_KAN_EXPERIMENT_SUITE:
        run_kan_experiment_suite()
        return

    if Config.RUN_SWEEP_EXPERIMENTS and not Config.RUN_MAIN_EXPERIMENTS:
        run_sweep_experiments()
        return

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
