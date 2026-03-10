import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import time
import warnings
from collections import OrderedDict
import re
import os
import hashlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# Подавляем FutureWarning для GradScaler
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# КОНФИГУРАЦИЯ С LSTM
# ============================================================================
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    CONTEXT_K = 20  # Большое окно для LSTM (81 символ → 162 признака)
    BATCH_SIZE = 65535  # MXNet-style large-block processing
    LEARNING_RATE = 3e-4 # More stable on small datasets
    WEIGHT_DECAY = 1e-4 # Adjusted for lighter model
    DROPOUT = 0.2 # Lower regularization for small files
    HIDDEN_DIM = 64 # Slightly higher capacity
    LSTM_LAYERS = 2
    LSTM_HIDDEN = 64 # Slightly higher capacity
    EPOCHS = 300
    PATIENCE = 12
    GRAD_CLIP = 0.6
    USE_NOISE_AUG = False
    NOISE_STD = 0.004 # Gentler augmentation
    USE_FOCAL_LOSS = False
    GAMMA = 1.5
    BER_LOSS_WEIGHT = 0.0  # Disable auxiliary BER term for numerical stability
    DATA_DIR_CANDIDATES = [
        Path("symbols_new"),
        Path("Symbols_1m_1ch_PR"),
        Path("."),
    ]
    BIDIRECTIONAL = True  # Bidirectional LSTM
    USE_ATTENTION = True  # Механизм внимания
    TRAIN_FILE_INDICES = None
    VAL_FILE_INDICES = None
    TEST_FILE_INDICES = None
    VAL_SPLIT_RATIO = 0.1
    TEST_SPLIT_RATIO = 0.1
    VALIDATE_EVERY = 5
    CHECKPOINT_EVERY = 50
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 8
    PERSISTENT_WORKERS = True
    MXNET_LARGE_BLOCK_MODE = True
    DROP_LAST_BLOCK = True
    MICRO_BATCH_SIZE = 4096  # Split large blocks into smaller CUDA-safe chunks
    USE_TORCH_COMPILE = True
    TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
    CHECKPOINT_DIR = Path("checkpoints")
    AUTO_RESUME = False
    DATA_CACHE_DIR = Path("data_cache")
    TRAIN_PROGRESS_EVERY = 250
    MAX_FILES: Optional[int] = 32
    ALIGN_SPLIT_WITH_MXNET_NOTEBOOK = True
    MXNET_NOTEBOOK_TRAIN_PORTION = 0.97
    MODEL_TYPES = ["lstm"]
    SAVE_COMPARISON_CSV = True

# Настройка для ускорения на CUDA
if Config.DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def env_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def is_main_process() -> bool:
    return env_rank() == 0

def log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

if is_main_process():
    log(f"✓ Device: {Config.DEVICE}")
    log(f"✓ Context window: ±{Config.CONTEXT_K} symbols (total {2*Config.CONTEXT_K+1})")
    log(f"✓ Focal Loss: {'ENABLED' if Config.USE_FOCAL_LOSS else 'disabled'}")

# ============================================================================
# CONSTELLATION (без изменений)
# ============================================================================
CONSTELLATION = torch.tensor([
    [-0.948683, -0.948683], [-0.948683, -0.316228], [-0.948683,  0.316228], [-0.948683,  0.948683],
    [-0.316228, -0.948683], [-0.316228, -0.316228], [-0.316228,  0.316228], [-0.316228,  0.948683],
    [ 0.316228, -0.948683], [ 0.316228, -0.316228], [ 0.316228,  0.316228], [ 0.316228,  0.948683],
    [ 0.948683, -0.948683], [ 0.948683, -0.316228], [ 0.948683,  0.316228], [ 0.948683,  0.948683],
], dtype=torch.float32)

BIT_LABELS = torch.tensor([
    [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,0,1,0],
    [0,1,0,0], [0,1,0,1], [0,1,1,1], [0,1,1,0],
    [1,1,0,0], [1,1,0,1], [1,1,1,1], [1,1,1,0],
    [1,0,0,0], [1,0,0,1], [1,0,1,1], [1,0,1,0]
], dtype=torch.uint8)

HAMMING_DISTANCES = (BIT_LABELS.unsqueeze(1) != BIT_LABELS.unsqueeze(0)).float().mean(dim=2)

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def current_device() -> torch.device:
    return Config.DEVICE

def mark_cudagraph_step_begin():
    if current_device().type != "cuda":
        return
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        compiler_mod.cudagraph_mark_step_begin()

def setup_distributed() -> Tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend="nccl")
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")

        Config.DEVICE = device
        return device, rank, world_size

    return Config.DEVICE, 0, 1

def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def reduce_scalar(value: float, op=dist.ReduceOp.SUM) -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor(value, device=current_device(), dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return tensor.item()

def bit_error_stats(tx_classes: torch.Tensor, rx_classes: torch.Tensor) -> Tuple[int, int]:
    device = current_device()
    bit_labels = BIT_LABELS.to(device)
    tx_bits = bit_labels[tx_classes.to(device)]
    rx_bits = bit_labels[rx_classes.to(device)]
    bit_errors = (tx_bits != rx_bits).sum().item()
    total_bits = tx_bits.numel()
    return bit_errors, total_bits

# ============================================================================
# МЕХАНИЗМ ВНИМАНИЯ ДЛЯ LSTM
# ============================================================================
class AttentionLayer(nn.Module):
    """
    Механизм внимания для выделения важных временных шагов в LSTM
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # lstm_output shape: (batch, seq_len, hidden_dim * num_directions)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = self.softmax(attention_weights)
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim * num_directions)
        return context

# ============================================================================
# LSTM АРХИТЕКТУРА
# ============================================================================
class LSTMRxEqualizer(nn.Module):
    """
    Улучшенная LSTM архитектура для обработки временных последовательностей:
    - Bi-directional LSTM для захвата контекста в обоих направлениях
    - Механизм внимания для выделения важных символов
    - Слой BatchNorm для стабилизации обучения
    """
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.seq_len = 2 * Config.CONTEXT_K + 1
        self.input_dim = input_dim

        # Embedding layer для предобработки
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5)
        )

        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=Config.LSTM_HIDDEN,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.LSTM_LAYERS > 1 else 0.0
        )

        # Механизм внимания (опционально)
        lstm_output_dim = Config.LSTM_HIDDEN * (2 if Config.BIDIRECTIONAL else 1)
        self.use_attention = Config.USE_ATTENTION
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_dim)

        self.center_fusion = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, lstm_output_dim),
            nn.LayerNorm(lstm_output_dim),
            nn.GELU()
        )

        # BatchNorm после LSTM
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        # Классификатор
        classifier_input_dim = lstm_output_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM // 2, 16)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                # Устанавливаем forget gate bias = 1
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[Config.LSTM_HIDDEN:2*Config.LSTM_HIDDEN] = 1.0

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input: (batch, seq_len * 2) -> (batch, seq_len, 2)
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.input_dim)

        # Embedding
        x = self.embedding(x)  # (batch, seq_len, 32)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim * num_directions)

        center_idx = self.seq_len // 2
        center_feature = lstm_out[:, center_idx, :]

        if self.use_attention:
            # Применяем механизм внимания
            context = self.attention(lstm_out)
        else:
            # Берем последний hidden state для bi-LSTM
            if Config.BIDIRECTIONAL:
                # Конкатенируем последние forward и backward states
                forward_hidden = hidden[-2, :, :]  # (batch, hidden_dim)
                backward_hidden = hidden[-1, :, :]  # (batch, hidden_dim)
                context = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim * 2)
            else:
                # Берем последний hidden state
                context = hidden[-1, :, :]  # (batch, hidden_dim)

        context = self.center_fusion(torch.cat([context, center_feature], dim=1))

        # Нормализация
        context = self.lstm_norm(context)

        # Классификация
        output = self.classifier(context)

        return output

# ============================================================================
# КОМБИНИРОВАННАЯ CNN-LSTM АРХИТЕКТУРА (альтернатива)
# ============================================================================
class HybridCNN_LSTM_Equalizer(nn.Module):
    """
    Гибридная архитектура CNN-LSTM:
    - CNN для извлечения локальных признаков
    - LSTM для временных зависимостей
    - Attention для объединения
    """
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.seq_len = 2 * Config.CONTEXT_K + 1

        # CNN часть для извлечения признаков
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
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

        # LSTM часть
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=Config.LSTM_HIDDEN,
            num_layers=2,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.DROPOUT
        )

        lstm_output_dim = Config.LSTM_HIDDEN * (2 if Config.BIDIRECTIONAL else 1)

        # Attention
        self.attention = AttentionLayer(lstm_output_dim)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, 16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Reshape и транспонирование для CNN
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

# ============================================================================
# MLP АРХИТЕКТУРА
# ============================================================================
class MLPRxEqualizer(nn.Module):
    """
    Архитектура MLP для обработки временных последовательностей.
    Принимает на вход полностью развернутое окно символов.
    """
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.seq_len = 2 * Config.CONTEXT_K + 1
        # The input to MLP is the flattened window (seq_len * input_dim)
        mlp_input_dim = self.seq_len * input_dim

        self.net = nn.Sequential(
            nn.Linear(mlp_input_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),

            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),

            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2), # Adjusted layer size for consistency
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5), # Half dropout for the last hidden layer

            nn.Linear(Config.HIDDEN_DIM // 2, 16) # Output 16 classes for QAM
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already flattened from WindowedSymbolDataset, so no explicit reshape needed here.
        # It's (batch_size, (2*K+1)*2)
        return self.net(x)

# ============================================================================
# ОСТАЛЬНЫЕ КОМПОНЕНТЫ (адаптированы для LSTM)
# ============================================================================
class WindowedSymbolDataset(Dataset):
    def __init__(self, rx_symbols: torch.Tensor, tx_classes: torch.Tensor,
                 k: int = 2, normalize: bool = True, noise_std: float = 0.0,
                 indices: Optional[torch.Tensor] = None,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 block_size: Optional[int] = None,
                 drop_last_block: bool = True):
        self.k = k
        self.noise_std = noise_std
        self.window_size = 2 * self.k + 1
        self.rx_symbols = rx_symbols.contiguous()
        self.tx_classes = tx_classes.to(torch.long).contiguous()

        num_windows = len(self.rx_symbols) - 2 * self.k
        if num_windows <= 0:
            raise ValueError("Not enough symbols for the selected context window")

        if indices is None:
            self.indices = torch.arange(num_windows, dtype=torch.long)
        else:
            self.indices = indices.to(torch.long).contiguous()

        self.block_size = block_size if block_size and block_size > 0 else None
        self.drop_last_block = drop_last_block

        if mean is not None and std is not None:
            self.mean = mean.clone()
            self.std = std.clone()
            self.std[self.std == 0] = 1.0
        elif normalize:
            self.mean = self.rx_symbols.mean(dim=0, keepdim=True)
            self.std = self.rx_symbols.std(dim=0, keepdim=True)
            self.std[self.std == 0] = 1.0
        else:
            self.mean = None
            self.std = None

        # Normalize once up front instead of redoing it in every __getitem__ call.
        if self.mean is not None and self.std is not None:
            self.rx_symbols = (self.rx_symbols - self.mean) / self.std
        self.rx_symbols = torch.nan_to_num(self.rx_symbols, nan=0.0, posinf=0.0, neginf=0.0)

        self.window_view = self.rx_symbols.unfold(0, self.window_size, 1).permute(0, 2, 1)

        self.training = False

    def __len__(self):
        if self.block_size is not None:
            if self.drop_last_block:
                return len(self.indices) // self.block_size
            return (len(self.indices) + self.block_size - 1) // self.block_size
        return len(self.indices)

    def __getitem__(self, idx):
        if self.block_size is not None:
            start = idx * self.block_size
            end = min(start + self.block_size, len(self.indices))
            block_indices = self.indices[start:end]
            x = self.window_view[block_indices]
            if self.noise_std > 0 and self.training:
                x = x + torch.randn_like(x) * self.noise_std
            y = self.tx_classes[block_indices + self.k]
            return x.reshape(x.size(0), -1), y

        sample_idx = int(self.indices[idx])
        x = self.window_view[sample_idx]
        if self.noise_std > 0 and self.training:
            # Добавляем шум только к I/Q компонентам
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        y = self.tx_classes[sample_idx + self.k]
        return x.reshape(-1), y

    def set_training(self, is_training: bool):
        self.training = is_training

class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset: Dataset):
        if not is_distributed():
            raise RuntimeError("DistributedEvalSampler requires initialized distributed process group")
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return (len(self.dataset) + self.world_size - 1 - self.rank) // self.world_size

def passthrough_collate(batch):
    # In large-block mode each dataset item is already a full training block.
    return batch[0]

def discover_symbol_files() -> Tuple[Path, List[int]]:
    pattern = re.compile(r"^(?:S_)?Symbols_1m_1ch_PR_(\d+)\.csv$", re.IGNORECASE)

    for candidate_dir in Config.DATA_DIR_CANDIDATES:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            continue

        file_indices = []
        for file_path in candidate_dir.iterdir():
            if not file_path.is_file():
                continue
            match = pattern.match(file_path.name)
            if match:
                file_indices.append(int(match.group(1)))

        if file_indices:
            file_indices = sorted(set(file_indices))
            if Config.MAX_FILES is not None:
                if Config.MAX_FILES < 3:
                    raise ValueError("Config.MAX_FILES must be at least 3")
                file_indices = file_indices[:Config.MAX_FILES]
            return candidate_dir, file_indices

    raise FileNotFoundError("No symbol CSV files found in any configured data directory")

def resolve_file_splits(available_indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
    if len(available_indices) < 3:
        raise ValueError("Need at least 3 files to build train/val/test splits")

    if Config.TRAIN_FILE_INDICES and Config.VAL_FILE_INDICES and Config.TEST_FILE_INDICES:
        train_indices = Config.TRAIN_FILE_INDICES
        val_indices = Config.VAL_FILE_INDICES
        test_indices = Config.TEST_FILE_INDICES
    elif Config.ALIGN_SPLIT_WITH_MXNET_NOTEBOOK:
        mx_train_files = max(1, int(len(available_indices) * Config.MXNET_NOTEBOOK_TRAIN_PORTION))
        mx_test_files = len(available_indices) - mx_train_files
        if mx_test_files < 1:
            mx_test_files = 1
            mx_train_files = len(available_indices) - 1

        mx_train_indices = available_indices[:mx_train_files]
        test_indices = available_indices[mx_train_files:]

        # Keep MXNet test files untouched; reserve one train file for validation.
        if len(mx_train_indices) < 2:
            raise ValueError(
                "MXNet-aligned split requires at least 2 train files (one for train and one for val)"
            )
        val_indices = [mx_train_indices[-1]]
        train_indices = mx_train_indices[:-1]
    else:
        total_files = len(available_indices)
        test_count = max(1, round(total_files * Config.TEST_SPLIT_RATIO))
        val_count = max(1, round(total_files * Config.VAL_SPLIT_RATIO))

        # Keep at least one training file even on small datasets.
        while total_files - test_count - val_count < 1:
            if test_count >= val_count and test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break

        train_end = total_files - val_count - test_count
        val_end = total_files - test_count

        train_indices = available_indices[:train_end]
        val_indices = available_indices[train_end:val_end]
        test_indices = available_indices[val_end:]

    missing = sorted(
        set(train_indices + val_indices + test_indices) - set(available_indices)
    )
    if missing:
        raise ValueError(f"Requested file indices are missing from dataset: {missing}")

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            f"Invalid split produced. train={train_indices}, val={val_indices}, test={test_indices}"
        )

    return train_indices, val_indices, test_indices

def load_all_files(file_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Загружает ВСЕ указанные файлы без пропусков."""
    tx_list, rx_list = [], []
    base_path, available_indices = discover_symbol_files()

    for idx in file_indices:
        # Ищем файл по нескольким возможным шаблонам
        candidates = [
            base_path / f"Symbols_1m_1ch_PR_{idx}.csv",
            base_path / f"S_symbols_1m_1ch_PR_{idx}.csv",
            Path(f"Symbols_1m_1ch_PR_{idx}.csv"),
            Path(f"S_symbols_1m_1ch_PR_{idx}.csv")
        ]

        file_path = None
        for cand in candidates:
            if cand.exists():
                file_path = cand
                break

        if file_path is None:
            raise FileNotFoundError(
                f"File for index {idx} not found in {base_path}. "
                f"Available indices: {available_indices[:10]}{'...' if len(available_indices) > 10 else ''}"
            )

        data = pd.read_csv(file_path, header=None, dtype=np.float32).to_numpy(copy=False)
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        if (nan_count > 0 or inf_count > 0) and is_main_process():
            log(f"  ⚠ File {idx}: found NaN={nan_count}, Inf={inf_count}; replacing with 0")
        # Guard against corrupted rows producing NaN/Inf in training.
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        tx_symbols = torch.from_numpy(data[:, 0:2].copy())
        rx_symbols = torch.from_numpy(data[:, 2:4].copy())
        tx_list.append(tx_symbols)
        rx_list.append(rx_symbols)
        if is_main_process():
            log(f"  ✓ File {idx}: {len(tx_symbols):,} symbols")

    tx_all = torch.cat(tx_list, dim=0)
    rx_all = torch.cat(rx_list, dim=0)
    if is_main_process():
        log(f"  → TOTAL: {len(tx_all):,} symbols from {len(file_indices)} files\n")
    return tx_all, rx_all

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Keep focal-loss math in fp32 to avoid NaNs/Inf under autocast fp16.
        inputs = inputs.float()
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        targets_one_hot = torch.zeros_like(probs, dtype=torch.float32).scatter_(1, targets.unsqueeze(1), 1.0)

        pt = (probs * targets_one_hot).sum(dim=1)
        pt = pt.clamp(min=1e-4, max=1.0 - 1e-4)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
            focal_weight = focal_weight * alpha_weight.sum(dim=1)

        loss = -focal_weight * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def symbols_to_classes(symbols: torch.Tensor) -> torch.Tensor:
    symbols = torch.nan_to_num(symbols, nan=0.0, posinf=0.0, neginf=0.0)
    constellation = CONSTELLATION.to(symbols.device)
    diff = symbols.unsqueeze(1) - constellation.unsqueeze(0)
    dist = torch.sum(diff ** 2, dim=2)
    return torch.argmin(dist, dim=1)

def calculate_ber_from_classes(tx_classes: torch.Tensor, rx_classes: torch.Tensor) -> float:
    device = current_device()
    bit_labels = BIT_LABELS.to(device)
    tx_bits = bit_labels[tx_classes.to(device)]
    rx_bits = bit_labels[rx_classes.to(device)]
    return (tx_bits != rx_bits).float().mean().item()

def ber_surrogate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    probs = torch.softmax(logits, dim=1)
    sample_costs = HAMMING_DISTANCES.to(logits.device)[targets.to(logits.device)]
    return (probs * sample_costs).sum(dim=1).mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_loss = (
            FocalLoss(gamma=Config.GAMMA)
            if Config.USE_FOCAL_LOSS
            else nn.CrossEntropyLoss(label_smoothing=0.1)
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cls_loss = self.classification_loss(inputs, targets)
        ber_loss = ber_surrogate_loss(inputs, targets)
        return cls_loss + Config.BER_LOSS_WEIGHT * ber_loss

def prepare_datasets(noise_std: float = Config.NOISE_STD) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    data_dir, available_indices = discover_symbol_files()
    train_file_indices, val_file_indices, test_file_indices = resolve_file_splits(available_indices)

    if is_main_process():
        log("="*80)
        log("PREPARING DATASETS FOR LSTM (FILE-LEVEL SPLIT)")
        log("="*80)
        log(f"Dataset directory: {data_dir}")
        log(f"Discovered files: {available_indices[0]}..{available_indices[-1]} ({len(available_indices)} total)")

    if is_main_process():
        log(f"\n[1/5] Loading TRAIN files {train_file_indices}...")
    tx_train, rx_train = cached_split_tensors("train", train_file_indices)

    if is_main_process():
        log(f"[2/5] Creating TRAIN windows (k={Config.CONTEXT_K} → {2*Config.CONTEXT_K+1} symbols)...")
    if len(tx_train) > 10000:
        tx_classes_list = [
            symbols_to_classes(s.to(current_device())).cpu()
            for s in tx_train.split(10000)
        ]
        tx_classes = torch.cat(tx_classes_list)
    else:
        tx_classes = symbols_to_classes(tx_train.to(current_device())).cpu()

    train_dataset = WindowedSymbolDataset(
        rx_train,
        tx_classes,
        k=Config.CONTEXT_K,
        normalize=True,
        noise_std=noise_std if Config.USE_NOISE_AUG else 0.0,
        block_size=Config.BATCH_SIZE if Config.MXNET_LARGE_BLOCK_MODE else None,
        drop_last_block=Config.DROP_LAST_BLOCK,
    )
    train_dataset.set_training(True)

    if is_main_process():
        log(f"[3/5] Loading VALIDATION files {val_file_indices}...")
    tx_val, rx_val = cached_split_tensors("val", val_file_indices)

    if is_main_process():
        log(f"[4/5] Creating VALIDATION windows with train normalization...")
    if len(tx_val) > 10000:
        tx_val_classes_list = [
            symbols_to_classes(s.to(current_device())).cpu()
            for s in tx_val.split(10000)
        ]
        tx_val_classes = torch.cat(tx_val_classes_list)
    else:
        tx_val_classes = symbols_to_classes(tx_val.to(current_device())).cpu()

    val_dataset = WindowedSymbolDataset(
        rx_val,
        tx_val_classes,
        k=Config.CONTEXT_K,
        normalize=False,
        noise_std=0.0,
        mean=train_dataset.mean,
        std=train_dataset.std,
        block_size=Config.BATCH_SIZE if Config.MXNET_LARGE_BLOCK_MODE else None,
        drop_last_block=Config.DROP_LAST_BLOCK,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedEvalSampler(val_dataset) if is_distributed() else None
    loader_batch_size = 1 if Config.MXNET_LARGE_BLOCK_MODE else Config.BATCH_SIZE
    collate_fn = passthrough_collate if Config.MXNET_LARGE_BLOCK_MODE else None
    train_loader = DataLoader(train_dataset, batch_size=loader_batch_size,
                             shuffle=train_sampler is None, sampler=train_sampler,
                             pin_memory=True,
                             num_workers=Config.NUM_WORKERS,
                             persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False,
                             prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
                             collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=loader_batch_size,
                           shuffle=False, sampler=val_sampler,
                           pin_memory=True,
                           num_workers=Config.NUM_WORKERS,
                           persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False,
                           prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
                           collate_fn=collate_fn)

    if is_main_process():
        log("\n" + "="*80)
        log("FINAL SPLIT FOR LSTM (FILE-LEVEL)")
        if Config.ALIGN_SPLIT_WITH_MXNET_NOTEBOOK:
            log(
                f"  Split policy: MXNet-aligned (train_portion={Config.MXNET_NOTEBOOK_TRAIN_PORTION:.2f}, "
                "test files kept identical to notebook)"
            )
        log(f"  Train files:   {train_file_indices}")
        log(f"  Val files:     {val_file_indices}")
        log(f"  Test files:    {test_file_indices} (will be loaded after training)")
        log(f"  Train windows: {len(train_dataset):,}")
        log(f"  Val windows:   {len(val_dataset):,}")
        log(f"  Sequence length: {2*Config.CONTEXT_K+1} symbols")
        log(f"  Input features per step: 2 (I/Q)")
        log("="*80 + "\n")

    return train_loader, val_loader, train_dataset.mean, train_dataset.std

def current_dataset_split() -> Tuple[Path, List[int], List[int], List[int]]:
    data_dir, available_indices = discover_symbol_files()
    train_file_indices, val_file_indices, test_file_indices = resolve_file_splits(available_indices)
    return data_dir, train_file_indices, val_file_indices, test_file_indices

def split_cache_path(split_name: str, data_dir: Path, file_indices: List[int]) -> Path:
    Config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(
        f"{data_dir.resolve()}::{split_name}::{','.join(map(str, file_indices))}".encode("utf-8")
    ).hexdigest()[:12]
    return Config.DATA_CACHE_DIR / f"{split_name}_{cache_key}.pt"

def cached_split_tensors(split_name: str, file_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    data_dir, _ = discover_symbol_files()
    cache_path = split_cache_path(split_name, data_dir, file_indices)

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        return cached["tx"], cached["rx"]

    if is_distributed() and not is_main_process():
        dist.barrier()
        cached = torch.load(cache_path, map_location="cpu")
        return cached["tx"], cached["rx"]

    tx_symbols, rx_symbols = load_all_files(file_indices)
    torch.save({"tx": tx_symbols, "rx": rx_symbols}, cache_path)
    if is_main_process():
        log(f"✓ Cached split '{split_name}' to {cache_path}")

    if is_distributed():
        dist.barrier()

    return tx_symbols, rx_symbols

def prepare_test_loader(mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    _, _, _, test_file_indices = current_dataset_split()

    if is_main_process():
        log("\n[TEST] Loading held-out test files after training...")
    tx_test, rx_test = cached_split_tensors("test", test_file_indices)
    tx_test_classes = symbols_to_classes(tx_test.to(current_device())).cpu()

    test_dataset = WindowedSymbolDataset(
        rx_test,
        tx_test_classes,
        k=Config.CONTEXT_K,
        normalize=False,
        mean=mean,
        std=std,
        block_size=Config.BATCH_SIZE if Config.MXNET_LARGE_BLOCK_MODE else None,
        drop_last_block=Config.DROP_LAST_BLOCK,
    )
    test_sampler = DistributedEvalSampler(test_dataset) if is_distributed() else None
    loader_batch_size = 1 if Config.MXNET_LARGE_BLOCK_MODE else Config.BATCH_SIZE
    collate_fn = passthrough_collate if Config.MXNET_LARGE_BLOCK_MODE else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
    )

    if is_main_process():
        log(f"[TEST] Test windows: {len(test_dataset):,}")

    return test_loader, tx_test, rx_test

def latest_checkpoint_path(model_name: str = "lstm") -> Optional[Path]:
    if not Config.CHECKPOINT_DIR.exists():
        return None

    checkpoint_paths = sorted(Config.CHECKPOINT_DIR.glob(f"{model_name}_epoch_*.pth"))
    if not checkpoint_paths:
        return None
    return checkpoint_paths[-1]

def checkpoint_matches_current_run(checkpoint: dict) -> bool:
    checkpoint_config = checkpoint.get("config", {})
    current_data_dir, train_file_indices, val_file_indices, test_file_indices = current_dataset_split()

    return (
        checkpoint_config.get("DATA_DIR") == str(current_data_dir)
        and checkpoint_config.get("TRAIN_FILE_INDICES") == train_file_indices
        and checkpoint_config.get("VAL_FILE_INDICES") == val_file_indices
        and checkpoint_config.get("TEST_FILE_INDICES") == test_file_indices
        and checkpoint_config.get("CONTEXT_K") == Config.CONTEXT_K
    )

def move_optimizer_to_device(optimizer: optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

def load_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=current_device())
    unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    move_optimizer_to_device(optimizer, current_device())
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint

def log_results_summary(all_results: List[dict]):
    if not is_main_process() or not all_results:
        return

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values("equalized_ber", ascending=True).reset_index(drop=True)

    log("\n" + "="*80)
    log("ARCHITECTURE COMPARISON")
    log("="*80)
    for row in summary_df.itertuples(index=False):
        log(
            f"{row.model_type.upper():>8} | "
            f"BER {row.equalized_ber:.6e} | "
            f"Acc {row.accuracy:.4%} | "
            f"RelImp {row.improvement_rel:.2f}% | "
            f"SER {row.ser:.6f}"
        )
    log("="*80)

    if Config.SAVE_COMPARISON_CSV:
        summary_path = Path("architecture_comparison.csv")
        summary_df.to_csv(summary_path, index=False)
        log(f"✓ Comparison saved to: {summary_path}")

def save_training_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    history: dict,
    epoch: int,
    model_name: str = "lstm",
    best_val_ber: Optional[float] = None,
    early_stopping_counter: int = 0,
    is_best: bool = False,
):
    if not is_main_process():
        return

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir, train_file_indices, val_file_indices, test_file_indices = current_dataset_split()
    checkpoint_path = Config.CHECKPOINT_DIR / f"{model_name}_epoch_{epoch:04d}.pth"

    checkpoint_payload = {
        'epoch': epoch,
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'normalization_mean': mean.cpu() if mean is not None else None,
        'normalization_std': std.cpu() if std is not None else None,
        'history': history,
        'best_val_ber': best_val_ber,
        'early_stopping_counter': early_stopping_counter,
        'config': {
            'CONTEXT_K': Config.CONTEXT_K,
            'HIDDEN_DIM': Config.HIDDEN_DIM,
            'LSTM_LAYERS': Config.LSTM_LAYERS,
            'LSTM_HIDDEN': Config.LSTM_HIDDEN,
            'DROPOUT': Config.DROPOUT,
            'LEARNING_RATE': Config.LEARNING_RATE,
            'WEIGHT_DECAY': Config.WEIGHT_DECAY,
            'NOISE_STD': Config.NOISE_STD,
            'DATA_DIR': str(data_dir),
            'TRAIN_FILE_INDICES': train_file_indices,
            'VAL_FILE_INDICES': val_file_indices,
            'TEST_FILE_INDICES': test_file_indices,
        },
        'model_architecture': unwrap_model(model).__class__.__name__,
    }

    torch.save(checkpoint_payload, checkpoint_path)
    log(f"✓ Checkpoint saved: {checkpoint_path}")

    if is_best:
        best_checkpoint_path = Config.CHECKPOINT_DIR / "best_checkpoint.pth"
        torch.save(checkpoint_payload, best_checkpoint_path)
        log(f"✓ Best checkpoint updated: {best_checkpoint_path}")

class EarlyStopping:
    def __init__(self, patience: int = Config.PATIENCE, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None
        self.last_improved = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        self.last_improved = False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.last_improved = True
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

    def restore_best_model(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

class Trainer:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 criterion=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion else CombinedLoss()
        self.scaler = torch.amp.GradScaler('cuda') if current_device().type == 'cuda' else None

    def train_epoch(self, loader: DataLoader, epoch: int = 0) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_start_time = time.time()
        num_batches = len(loader)

        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        for batch_idx, (inputs, labels) in enumerate(loader, start=1):
            inputs = inputs.to(current_device(), non_blocking=True)
            labels = labels.to(current_device(), non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            batch_size = inputs.size(0)
            micro_batch = Config.MICRO_BATCH_SIZE if Config.MICRO_BATCH_SIZE and Config.MICRO_BATCH_SIZE > 0 else batch_size
            batch_loss_accum = 0.0

            if current_device().type == 'cuda' and self.scaler is not None:
                skip_batch = False
                if micro_batch < batch_size:
                    for start in range(0, batch_size, micro_batch):
                        end = min(start + micro_batch, batch_size)
                        chunk_inputs = inputs[start:end]
                        chunk_labels = labels[start:end]
                        chunk_weight = (end - start) / batch_size
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            mark_cudagraph_step_begin()
                            chunk_outputs = self.model(chunk_inputs)
                        chunk_loss = self.criterion(chunk_outputs.float(), chunk_labels)
                        if not torch.isfinite(chunk_loss):
                            if is_main_process():
                                log(f"⚠ Non-finite loss at epoch {epoch+1}, batch {batch_idx}; skipping optimizer step")
                            skip_batch = True
                            break
                        self.scaler.scale(chunk_loss * chunk_weight).backward()
                        batch_loss_accum += chunk_loss.item() * (end - start)
                else:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        mark_cudagraph_step_begin()
                        outputs = self.model(inputs)
                    chunk_loss = self.criterion(outputs.float(), labels)
                    if not torch.isfinite(chunk_loss):
                        if is_main_process():
                            log(f"⚠ Non-finite loss at epoch {epoch+1}, batch {batch_idx}; skipping optimizer step")
                        skip_batch = True
                    if not skip_batch:
                        self.scaler.scale(chunk_loss).backward()
                        batch_loss_accum = chunk_loss.item() * batch_size

                if skip_batch:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                skip_batch = False
                if micro_batch < batch_size:
                    for start in range(0, batch_size, micro_batch):
                        end = min(start + micro_batch, batch_size)
                        chunk_inputs = inputs[start:end]
                        chunk_labels = labels[start:end]
                        chunk_weight = (end - start) / batch_size
                        mark_cudagraph_step_begin()
                        chunk_outputs = self.model(chunk_inputs)
                        chunk_loss = self.criterion(chunk_outputs.float(), chunk_labels)
                        if not torch.isfinite(chunk_loss):
                            if is_main_process():
                                log(f"⚠ Non-finite loss at epoch {epoch+1}, batch {batch_idx}; skipping optimizer step")
                            skip_batch = True
                            break
                        (chunk_loss * chunk_weight).backward()
                        batch_loss_accum += chunk_loss.item() * (end - start)
                else:
                    mark_cudagraph_step_begin()
                    outputs = self.model(inputs)
                    chunk_loss = self.criterion(outputs.float(), labels)
                    if not torch.isfinite(chunk_loss):
                        if is_main_process():
                            log(f"⚠ Non-finite loss at epoch {epoch+1}, batch {batch_idx}; skipping optimizer step")
                        skip_batch = True
                    if not skip_batch:
                        chunk_loss.backward()
                        batch_loss_accum = chunk_loss.item() * batch_size

                if skip_batch:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
                self.optimizer.step()

            total_loss += batch_loss_accum
            total_samples += batch_size

            if (
                is_main_process()
                and Config.TRAIN_PROGRESS_EVERY > 0
                and (batch_idx % Config.TRAIN_PROGRESS_EVERY == 0 or batch_idx == num_batches)
            ):
                elapsed = time.time() - epoch_start_time
                running_speed = total_samples / max(elapsed, 1e-8)
                log(
                    f"  Epoch {epoch+1:4d} progress | "
                    f"Batch {batch_idx:5d}/{num_batches} | "
                    f"TrainLoss {total_loss / max(total_samples, 1):.6f} | "
                    f"Speed {running_speed:,.0f} samp/s | "
                    f"Elapsed {elapsed/60:.1f} min"
                )

        if is_distributed():
            total_loss = reduce_scalar(total_loss)
            total_samples = reduce_scalar(total_samples)

        epoch_time = time.time() - epoch_start_time
        samples_per_sec = total_samples / max(epoch_time, 1e-8)
        return total_loss / max(total_samples, 1), epoch_time, samples_per_sec

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        bit_errors = 0
        total_bits = 0

        for inputs, labels in loader:
            inputs = inputs.to(current_device(), non_blocking=True)
            labels = labels.to(current_device(), non_blocking=True)
            batch_size = labels.size(0)
            micro_batch = Config.MICRO_BATCH_SIZE if Config.MICRO_BATCH_SIZE and Config.MICRO_BATCH_SIZE > 0 else batch_size

            for start in range(0, batch_size, micro_batch):
                end = min(start + micro_batch, batch_size)
                chunk_inputs = inputs[start:end]
                chunk_labels = labels[start:end]

                if current_device().type == 'cuda' and self.scaler is not None:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        mark_cudagraph_step_begin()
                        outputs = self.model(chunk_inputs)
                    loss = self.criterion(outputs.float(), chunk_labels)
                else:
                    mark_cudagraph_step_begin()
                    outputs = self.model(chunk_inputs)
                    loss = self.criterion(outputs.float(), chunk_labels)

                total_loss += loss.item() * chunk_inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == chunk_labels).sum().item()
                total += chunk_labels.size(0)
                sample_bit_errors, sample_total_bits = bit_error_stats(chunk_labels, preds)
                bit_errors += sample_bit_errors
                total_bits += sample_total_bits

        if is_distributed():
            total_loss = reduce_scalar(total_loss)
            correct = reduce_scalar(correct)
            total = reduce_scalar(total)
            bit_errors = reduce_scalar(bit_errors)
            total_bits = reduce_scalar(total_bits)

        val_loss = total_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        val_ber = bit_errors / max(total_bits, 1)
        return val_loss, val_acc, val_ber

def train_equalizer(train_loader: DataLoader, val_loader: DataLoader,
                   model_type: str = "lstm",
                   mean: Optional[torch.Tensor] = None,
                   std: Optional[torch.Tensor] = None) -> Tuple[nn.Module, dict]:
    """
    Обучает равнизатор

    Args:
        model_type: "lstm", "hybrid", "cnn_lstm" or "mlp"
    """
    if is_main_process():
        log(f"\nCreating {model_type.upper()} model...")
        log(f"  DEBUG - Config.HIDDEN_DIM: {Config.HIDDEN_DIM}")
        log(f"  DEBUG - Config.LSTM_HIDDEN: {Config.LSTM_HIDDEN}")
        log(f"  DEBUG - Config.LEARNING_RATE: {Config.LEARNING_RATE}")
        log(f"  DEBUG - Config.WEIGHT_DECAY: {Config.WEIGHT_DECAY}")
        log(f"  DEBUG - Config.DROPOUT: {Config.DROPOUT}")

    base_model: nn.Module
    if model_type == "lstm":
        base_model = LSTMRxEqualizer(input_dim=2).to(current_device())
        if is_main_process():
            log(f"✓ Architecture: LSTM (layers={Config.LSTM_LAYERS}, hidden={Config.LSTM_HIDDEN}, bidirectional={Config.BIDIRECTIONAL}, attention={Config.USE_ATTENTION})")
    elif model_type == "hybrid" or model_type == "cnn_lstm":
        base_model = HybridCNN_LSTM_Equalizer(input_dim=2).to(current_device())
        if is_main_process():
            log(f"✓ Architecture: Hybrid CNN-LSTM (LSTM layers=2, hidden={Config.LSTM_HIDDEN}, bidirectional={Config.BIDIRECTIONAL})")
    elif model_type == "mlp": # Added MLP model
        base_model = MLPRxEqualizer(input_dim=2).to(current_device())
        if is_main_process():
            log(f"✓ Architecture: MLP (hidden={Config.HIDDEN_DIM})")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if Config.USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            base_model = torch.compile(base_model, mode=Config.TORCH_COMPILE_MODE)
            if is_main_process():
                log(f"✓ torch.compile enabled (mode={Config.TORCH_COMPILE_MODE})")
        except Exception as compile_error:
            if is_main_process():
                log(f"⚠ torch.compile disabled: {compile_error}")

    model = DDP(base_model, device_ids=[current_device().index]) if is_distributed() and current_device().type == "cuda" else base_model

    # Подсчет параметров
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if is_main_process():
        log(f"  Total parameters: {total_params:,}")
        log(f"  Trainable parameters: {trainable_params:,}")

    adamw_kwargs = dict(lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    if current_device().type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    criterion = CombinedLoss()

    trainer = Trainer(model, optimizer, scheduler, criterion)
    early_stopping = EarlyStopping(patience=Config.PATIENCE)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_ber': [],
        'val_epochs': [],
        'lr': [],
        'epoch_time_sec': [],
        'train_samples_per_sec': [],
    }
    start_epoch = 0

    if Config.AUTO_RESUME:
        checkpoint_path = latest_checkpoint_path(model_type)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=current_device())
            if checkpoint_matches_current_run(checkpoint):
                unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                move_optimizer_to_device(optimizer, current_device())
                if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                history = checkpoint.get('history', history)
                history.setdefault('epoch_time_sec', [])
                history.setdefault('train_samples_per_sec', [])
                early_stopping.best_loss = checkpoint.get('best_val_ber', early_stopping.best_loss)
                early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
                if is_main_process():
                    log(f"✓ Resumed from checkpoint: {checkpoint_path} (epoch {start_epoch})")
            elif is_main_process():
                log(f"✓ Skipping incompatible checkpoint: {checkpoint_path}")
    if is_main_process():
        log(f"\nSTARTING TRAINING {model_type.upper()} on {current_device().type.upper()}...")
        train_batches_per_epoch = len(train_loader)
        val_batches = len(val_loader)
        log(
            f"  Train batches/epoch: {train_batches_per_epoch:,} | "
            f"Val batches: {val_batches:,} | "
            f"Validate every {Config.VALIDATE_EVERY} epochs"
        )
        if start_epoch > 0:
            next_val_epoch = start_epoch + 1
            while next_val_epoch <= Config.EPOCHS and next_val_epoch % Config.VALIDATE_EVERY != 0:
                next_val_epoch += 1
            if next_val_epoch <= Config.EPOCHS:
                log(f"  Resume note: next validation log will be at epoch {next_val_epoch}")
        log("="*80)

    start_time = time.time()

    for epoch in range(start_epoch, Config.EPOCHS):
        train_loss, epoch_time_sec, train_samples_per_sec = trainer.train_epoch(train_loader, epoch)
        should_validate = (
            epoch == 0 or
            (epoch + 1) % Config.VALIDATE_EVERY == 0 or
            epoch == Config.EPOCHS - 1
        )

        if trainer.scheduler is not None:
            trainer.scheduler.step()

        if not should_validate:
            if is_main_process():
                history['train_loss'].append(train_loss)
                history['lr'].append(optimizer.param_groups[0]['lr'])
                history['epoch_time_sec'].append(epoch_time_sec)
                history['train_samples_per_sec'].append(train_samples_per_sec)
                log(f"Epoch {epoch+1:4d}/{Config.EPOCHS} | "
                      f"Train: {train_loss:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"Speed: {train_samples_per_sec:,.0f} samp/s | "
                      f"Time: {epoch_time_sec/60:.1f} min | "
                      f"Val: skipped")
            continue

        val_loss, val_acc, val_ber = trainer.validate(val_loader)

        stop_training = False
        if is_main_process():
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_ber'].append(val_ber)
            history['val_epochs'].append(epoch + 1)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time_sec'].append(epoch_time_sec)
            history['train_samples_per_sec'].append(train_samples_per_sec)

            if early_stopping(val_ber, unwrap_model(model)):
                log(f"\n✓ Early stopping at epoch {epoch+1}")
                stop_training = True

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == Config.EPOCHS - 1:
                improvement = ((history['val_loss'][0] - val_loss) / history['val_loss'][0]) * 100 if len(history['val_loss']) > 1 else 0
                log(f"Epoch {epoch+1:4d}/{Config.EPOCHS} | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"Acc: {val_acc:.4f} | BER: {val_ber:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"Speed: {train_samples_per_sec:,.0f} samp/s | "
                      f"Time: {epoch_time_sec:.1f}s | "
                      f"\u0394: {improvement:+.1f}%")

            if (epoch + 1) % Config.CHECKPOINT_EVERY == 0:
                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    mean=mean,
                    std=std,
                    history=history,
                    epoch=epoch + 1,
                    model_name=model_type,
                    best_val_ber=early_stopping.best_loss,
                    early_stopping_counter=early_stopping.counter,
                    )

            if early_stopping.last_improved:
                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    mean=mean,
                    std=std,
                    history=history,
                    epoch=epoch + 1,
                    model_name=model_type,
                    best_val_ber=early_stopping.best_loss,
                    early_stopping_counter=early_stopping.counter,
                    is_best=True,
                )

        if is_distributed():
            stop_tensor = torch.tensor(int(stop_training), device=current_device())
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

    if is_main_process():
        early_stopping.restore_best_model(unwrap_model(model))
    if is_distributed():
        dist.barrier()
        state_dict = unwrap_model(model).state_dict() if is_main_process() else None
        obj_list = [state_dict]
        dist.broadcast_object_list(obj_list, src=0)
        if not is_main_process():
            unwrap_model(model).load_state_dict(obj_list[0])

    training_time = (time.time() - start_time) / 60
    if is_main_process():
        log(f"✓ Training completed in {training_time:.1f} min | Best val BER: {early_stopping.best_loss:.6f}\n")

    return unwrap_model(model), history

@torch.no_grad()
def comprehensive_evaluation(model: nn.Module, test_loader: DataLoader,
                           tx_symbols: torch.Tensor, rx_symbols: torch.Tensor) -> dict:
    model.eval()

    # Baseline BER
    baseline_ber = calculate_ber_from_classes(
        symbols_to_classes(tx_symbols.to(current_device())),
        symbols_to_classes(rx_symbols.to(current_device()))
    )

    # Model predictions
    correct = 0
    total = 0
    bit_errors = 0
    total_bits = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(current_device(), non_blocking=True)
        labels = labels.to(current_device(), non_blocking=True)
        batch_size = labels.size(0)
        micro_batch = Config.MICRO_BATCH_SIZE if Config.MICRO_BATCH_SIZE and Config.MICRO_BATCH_SIZE > 0 else batch_size

        for start in range(0, batch_size, micro_batch):
            end = min(start + micro_batch, batch_size)
            chunk_inputs = inputs[start:end]
            chunk_labels = labels[start:end]
            if current_device().type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    mark_cudagraph_step_begin()
                    logits = model(chunk_inputs)
            else:
                mark_cudagraph_step_begin()
                logits = model(chunk_inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == chunk_labels).sum().item()
            total += chunk_labels.size(0)
            sample_bit_errors, sample_total_bits = bit_error_stats(chunk_labels, preds)
            bit_errors += sample_bit_errors
            total_bits += sample_total_bits

    if is_distributed():
        correct = reduce_scalar(correct)
        total = reduce_scalar(total)
        bit_errors = reduce_scalar(bit_errors)
        total_bits = reduce_scalar(total_bits)

    equalized_ber = bit_errors / total_bits
    ser = 1 - (correct / total)
    accuracy = correct / total

    return {
        'baseline_ber': baseline_ber,
        'equalized_ber': equalized_ber,
        'ser': ser,
        'accuracy': accuracy,
        'improvement_abs': baseline_ber - equalized_ber,
        'improvement_rel': (1 - equalized_ber / baseline_ber) * 100,
        'improvement_db': 10 * np.log10(baseline_ber / equalized_ber) if equalized_ber > 0 else float('inf')
    }

def plot_results(history: dict, eval_results: dict, model_name: str = "LSTM"):
    if not is_main_process():
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    train_epochs = list(range(1, len(history['train_loss']) + 1))
    val_epochs = history.get('val_epochs', list(range(1, len(history['val_loss']) + 1)))

    # Loss curves
    axes[0,0].plot(train_epochs, history['train_loss'], label='Train', linewidth=2, alpha=0.8)
    axes[0,0].plot(val_epochs, history['val_loss'], label='Val', linewidth=2)
    axes[0,0].set_title(f'{model_name} - Loss Curves', fontweight='bold')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    axes[0,0].set_yscale('log')

    # Accuracy
    axes[0,1].plot(val_epochs, history['val_acc'], color='green', linewidth=2)
    axes[0,1].set_title(f'{model_name} - Validation Accuracy', fontweight='bold')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Accuracy')
    axes[0,1].grid(alpha=0.3)
    axes[0,1].set_ylim([0, 1])

    # LR schedule
    axes[0,2].plot(train_epochs[:len(history['lr'])], history['lr'], color='red', linewidth=2)
    axes[0,2].set_yscale('log')
    axes[0,2].set_title('Learning Rate Schedule', fontweight='bold')
    axes[0,2].set_xlabel('Epoch'); axes[0,2].set_ylabel('LR')
    axes[0,2].grid(alpha=0.3)

    # BER comparison
    axes[1,0].bar(['Baseline', f'{model_name} EQ'],
                  [eval_results['baseline_ber'], eval_results['equalized_ber']],
                  color=['#ff7675', '#55efc4'], edgecolor='black', linewidth=1.5)
    axes[1,0].set_yscale('log')
    axes[1,0].set_title('BER Comparison (log scale)', fontweight='bold')
    axes[1,0].set_ylabel('BER'); axes[1,0].grid(axis='y', alpha=0.3)

    # Improvement metrics
    metrics = [
        'Abs Reduction\n(pp)', 'Rel Improvement\n(%)', 'SNR Gain\n(dB)'
    ]
    values = [
        eval_results['improvement_abs']*100,
        eval_results['improvement_rel'],
        min(eval_results['improvement_db'], 20)  # Ограничиваем для визуализации
    ]

    axes[1,1].bar(metrics, values,
                  color=['#74b9ff', '#a29bfe', '#fd79a8'], edgecolor='black', linewidth=1.5)
    axes[1,1].set_title('Improvement Metrics', fontweight='bold')
    axes[1,1].set_ylabel('Value'); axes[1,1].grid(axis='y', alpha=0.3)

    # Training dynamics
    axes[1,2].plot(train_epochs, [l*100 for l in history['train_loss']],
                   label='Train Loss ×100', alpha=0.7, linewidth=1)
    axes[1,2].plot(val_epochs, [a*100 for a in history['val_acc']],
                   label='Val Acc (%)', alpha=0.7, linewidth=2)
    axes[1,2].set_title('Training Dynamics', fontweight='bold')
    axes[1,2].set_xlabel('Epoch'); axes[1,2].set_ylabel('Value')
    axes[1,2].legend(); axes[1,2].grid(alpha=0.3)

    plt.suptitle(f'{model_name} Equalizer Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"ber_results_{model_name.lower()}.png", dpi=150, bbox_inches='tight') # Dynamic filename
    print(f"✓ Results saved to ber_results_{model_name.lower()}.png")

def save_model(model: nn.Module, mean: torch.Tensor, std: torch.Tensor,
               model_name: str = "lstm", path: str = None, config_override: Optional[dict] = None):
    if not is_main_process():
        return

    if path is None:
        path = f"ber_equalizer_{model_name}_final.pth"

    data_dir, train_file_indices, val_file_indices, test_file_indices = current_dataset_split()

    # Use config_override if provided, otherwise use current Config values
    current_config = {
        'CONTEXT_K': Config.CONTEXT_K,
        'HIDDEN_DIM': Config.HIDDEN_DIM,
        'DROPOUT': Config.DROPOUT,
        'LSTM_LAYERS': Config.LSTM_LAYERS,
        'LSTM_HIDDEN': Config.LSTM_HIDDEN,
        'BIDIRECTIONAL': Config.BIDIRECTIONAL,
        'USE_ATTENTION': Config.USE_ATTENTION,
        'USE_FOCAL_LOSS': Config.USE_FOCAL_LOSS,
        'GAMMA': Config.GAMMA if Config.USE_FOCAL_LOSS else None,
        'LEARNING_RATE': Config.LEARNING_RATE,
        'WEIGHT_DECAY': Config.WEIGHT_DECAY,
        'NOISE_STD': Config.NOISE_STD,
        'PATIENCE': Config.PATIENCE,
        'DATA_DIR': str(data_dir),
        'TRAIN_FILE_INDICES': train_file_indices,
        'VAL_FILE_INDICES': val_file_indices,
        'TEST_FILE_INDICES': test_file_indices,
    }

    # If model_name is mlp, ensure LSTM-specific configs are not saved
    if model_name == "mlp":
        current_config.pop('LSTM_LAYERS', None)
        current_config.pop('LSTM_HIDDEN', None)
        current_config.pop('BIDIRECTIONAL', None)
        current_config.pop('USE_ATTENTION', None)

    final_config = config_override if config_override else current_config


    torch.save({
        'model_state_dict': unwrap_model(model).state_dict(),
        'normalization_mean': mean.cpu() if mean is not None else None,
        'normalization_std': std.cpu() if std is not None else None,
        'context_k': Config.CONTEXT_K,
        'input_dim': 2,
        'seq_len': 2 * Config.CONTEXT_K + 1,
        'config': final_config,
        'model_architecture': model.__class__.__name__
    }, path)
    print(f"✓ Model saved to: {path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    device, rank, world_size = setup_distributed()

    try:
        if is_main_process():
            print(f"✓ Runtime device: {device}")
            if world_size > 1:
                print(f"✓ Distributed training enabled on {world_size} GPUs")

        data_dir, train_file_indices, val_file_indices, test_file_indices = current_dataset_split()

        train_loader, val_loader, mean, std = prepare_datasets(noise_std=Config.NOISE_STD)

        all_results = []
        test_loader = None
        tx_test_full = None
        rx_test_full = None

        for model_type in Config.MODEL_TYPES:
            model, history = train_equalizer(
                train_loader,
                val_loader,
                model_type=model_type,
                mean=mean,
                std=std,
            )

            if test_loader is None:
                test_loader, tx_test_full, rx_test_full = prepare_test_loader(mean, std)

            results = comprehensive_evaluation(model, test_loader, tx_test_full, rx_test_full)
            results["model_type"] = model_type
            all_results.append(results)

            if is_main_process():
                log("\n" + "="*80)
                log(f"FINAL EVALUATION - {model_type.upper()} (test files {test_file_indices} from {data_dir})")
                log("="*80)
                log(f"Baseline BER:          {results['baseline_ber']:.6e}")
                log(f"Equalized BER:         {results['equalized_ber']:.6e}")
                log(f"Symbol Accuracy:       {results['accuracy']:.4%}")
                log(f"Absolute reduction:    {results['improvement_abs']*100:.4f} pp")
                log(f"Relative improvement:  {results['improvement_rel']:.2f}%")
                log(f"SNR gain:              {results['improvement_db']:.2f} dB")
                log(f"Symbol Error Rate:     {results['ser']:.6f}")
                log("="*80)

            plot_results(history, results, model_name=model_type)
            save_model(model, mean, std, model_name=model_type)
            if current_device().type == "cuda":
                model.to("cpu")
                del model
                torch.cuda.empty_cache()

        log_results_summary(all_results)

        if is_main_process():
            log(f"\n✓ Training complete for models: {', '.join(m.upper() for m in Config.MODEL_TYPES)}")
            if world_size > 1:
                log("\nRun with: torchrun --nproc_per_node=4 experiment1.py")
    finally:
        cleanup_distributed()
