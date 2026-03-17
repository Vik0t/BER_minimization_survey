import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class Config:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    DATA_DIR_CANDIDATES = [Path("symbols_new"), Path("Symbols_1m_1ch_PR"), Path(".")]
    MAX_FILES = 32
    TRAIN_PORTION = 0.97  # MXNet notebook split: last train file goes to val, rest to test.

    CONTEXT_K = 20
    SEQ_LEN = 2 * CONTEXT_K + 1
    INPUT_DIM = 2

    HIDDEN_DIM = 64
    DROPOUT = 0.2
    LSTM_HIDDEN = 64
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True
    USE_ATTENTION = True

    EPOCHS = 250
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    TRAIN_BLOCK_SIZE = 131072
    EVAL_BATCH_SIZE = 131072
    MIN_BLOCK_SIZE = 4096
    USE_AMP = True
    USE_TORCH_COMPILE = False
    TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"

    DECAY_STEPS = 20
    MIN_LR = 1e-4
    LOG_EVERY = 5
    TEST_BER_EVERY = 10

    OUT_DIR = Path("clean_compare_outputs")
    MODEL_TYPES = ["lstm", "cnn_lstm", "mlp"]
    SAVE_BEST = True


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


def prepare_data() -> Dict[str, torch.Tensor]:
    base_dir, all_indices = discover_symbol_files()
    train_idx, val_idx, test_idx = resolve_splits(all_indices)
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
def evaluate_split(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[float, float, float]:
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
    test_loss, test_acc, test_ber = evaluate_split(model, data["test_x"], data["test_y"], eval_batch_size)
    return {
        "baseline_ber": baseline_ber,
        "equalized_ber": test_ber,
        "accuracy": test_acc,
        "ser": 1.0 - test_acc,
        "test_loss": test_loss,
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

    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP)
    best_train_loss = float("inf")
    steps_wo_min = 0
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
        order = torch.randperm(train_x.size(0))
        shuffled_x = train_x.index_select(0, order)
        shuffled_y = train_y.index_select(0, order)
        start = 0
        while start < shuffled_x.size(0):
            end = min(start + train_block_size, shuffled_x.size(0))
            xb = shuffled_x[start:end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
            yb = shuffled_y[start:end].to(Config.DEVICE, non_blocking=Config.DEVICE.type == "cuda")
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

        val_loss, val_acc, val_ber = evaluate_split(model, data["val_x"], data["val_y"], eval_batch_size)
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
            steps_wo_min = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if Config.SAVE_BEST:
                torch.save(best_state, Config.OUT_DIR / f"{model_name}_best.pth")
        else:
            steps_wo_min += 1

        if steps_wo_min >= Config.DECAY_STEPS:
            steps_wo_min = 0
            new_lr = optimizer.param_groups[0]["lr"] / 2.0
            for group in optimizer.param_groups:
                group["lr"] = new_lr
            log(f"{model_name} | epoch {epoch+1} -- Learning rate {new_lr:.6g}")
            if new_lr < Config.MIN_LR:
                log(f"{model_name} | train finished: lr < {Config.MIN_LR}")
                break

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


if __name__ == "__main__":
    main()
