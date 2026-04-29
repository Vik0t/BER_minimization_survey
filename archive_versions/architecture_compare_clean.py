import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class Config:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    DATA_DIR_CANDIDATES = [Path("symbols_new"), Path("Symbols_1m_1ch_PR"), Path(".")]
    MAX_FILES = 32
    TRAIN_PORTION = 0.97  # MXNet notebook style

    CONTEXT_K = 20
    SEQ_LEN = 2 * CONTEXT_K + 1
    INPUT_DIM = 2

    LSTM_HIDDEN = 64
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    HIDDEN_DIM = 64
    DROPOUT = 0.2

    EPOCHS = 150  # MXNet default
    LEARNING_RATE = 1e-3  # MXNet default
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32768
    MICRO_BATCH_SIZE = 8192
    NUM_WORKERS = 8
    PIN_MEMORY = False
    USE_AMP = True
    USE_TORCH_COMPILE = True
    TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
    BER_LOSS_WEIGHT = 0.05
    LABEL_SMOOTHING = 0.05
    USE_EMA = True
    EMA_DECAY = 0.999
    WARMUP_EPOCHS = 3

    DECAY_STEPS = 20  # MXNet style LR decay trigger
    MIN_LR = 1e-4
    LOG_EVERY = 5

    OUT_DIR = Path("clean_compare_outputs")
    MODELS = ["lstm", "cnn_lstm"]
    SAVE_BEST = True

    MXNET_REF_BER_AFTER = 0.004768076632561972
    MXNET_REF_EPOCH_SEC = 140.11567759513855 / 50.0


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
HAMMING_DISTANCES = (BIT_LABELS.unsqueeze(1) != BIT_LABELS.unsqueeze(0)).float().mean(dim=2)


def log(msg: str):
    print(msg, flush=True)


def mark_cudagraph_step_begin():
    if Config.DEVICE.type != "cuda":
        return
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        compiler_mod.cudagraph_mark_step_begin()


def iter_micro_batches(x: torch.Tensor, y: torch.Tensor):
    micro = Config.MICRO_BATCH_SIZE if Config.MICRO_BATCH_SIZE and Config.MICRO_BATCH_SIZE > 0 else x.size(0)
    for start in range(0, x.size(0), micro):
        end = min(start + micro, x.size(0))
        yield x[start:end], y[start:end], (end - start) / x.size(0)


class EMAHelper:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        state = model.state_dict()
        for k in self.shadow:
            # BatchNorm counters and other integer/bool buffers must not be EMA-updated.
            if self.shadow[k].is_floating_point() or self.shadow[k].is_complex():
                self.shadow[k].mul_(self.decay).add_(state[k], alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(state[k])

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def combined_ber_loss(logits: torch.Tensor, targets: torch.Tensor, ce: nn.Module) -> torch.Tensor:
    logits_fp32 = logits.float()
    ce_loss = ce(logits_fp32, targets)
    probs = torch.softmax(logits_fp32, dim=1)
    costs = HAMMING_DISTANCES.to(logits.device)[targets]
    ber_loss = (probs * costs).sum(dim=1).mean()
    return ce_loss + Config.BER_LOSS_WEIGHT * ber_loss


def symbols_to_classes(symbols: torch.Tensor) -> torch.Tensor:
    constellation = CONSTELLATION.to(symbols.device)
    diff = symbols.unsqueeze(1) - constellation.unsqueeze(0)
    dist = torch.sum(diff**2, dim=2)
    return torch.argmin(dist, dim=1)


def calculate_ber_from_classes(tx_classes: torch.Tensor, rx_classes: torch.Tensor) -> float:
    device = tx_classes.device
    bit_labels = BIT_LABELS.to(device)
    tx_bits = bit_labels[tx_classes]
    rx_bits = bit_labels[rx_classes]
    return (tx_bits != rx_bits).float().mean().item()


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
        seq_len = Config.SEQ_LEN
        input_dim = Config.INPUT_DIM
        lstm_output_dim = Config.LSTM_HIDDEN * (2 if Config.BIDIRECTIONAL else 1)
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.use_attention = Config.USE_ATTENTION

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 32),
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
        self.attention = AttentionLayer(lstm_output_dim)
        self.center_fusion = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, lstm_output_dim),
            nn.LayerNorm(lstm_output_dim),
            nn.GELU(),
        )
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT * 0.5),
            nn.Linear(Config.HIDDEN_DIM // 2, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.seq_len, self.input_dim)
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        center_feature = lstm_out[:, self.seq_len // 2, :]
        if self.use_attention:
            context = self.attention(lstm_out)
        else:
            if Config.BIDIRECTIONAL:
                context = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                context = hidden[-1]
        context = self.center_fusion(torch.cat([context, center_feature], dim=1))
        context = self.lstm_norm(context)
        return self.classifier(context)


class HybridCNNLSTMEqualizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = Config.SEQ_LEN
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
            nn.Linear(Config.HIDDEN_DIM, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.seq_len, 2).transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.attention(x)
        return self.classifier(x)


class WindowedSymbolDataset(Dataset):
    def __init__(self, rx_symbols: torch.Tensor, tx_classes: torch.Tensor, k: int, mean=None, std=None):
        self.k = k
        self.window_size = 2 * k + 1
        self.rx_symbols = rx_symbols.contiguous()
        self.tx_classes = tx_classes.to(torch.long).contiguous()

        if mean is None or std is None:
            self.mean = self.rx_symbols.mean(dim=0, keepdim=True)
            self.std = self.rx_symbols.std(dim=0, keepdim=True)
            self.std[self.std == 0] = 1.0
        else:
            self.mean = mean
            self.std = std

        self.rx_symbols = (self.rx_symbols - self.mean) / self.std
        self.window_view = self.rx_symbols.unfold(0, self.window_size, 1).permute(0, 2, 1)

    def __len__(self):
        return self.window_view.size(0)

    def __getitem__(self, idx):
        x = self.window_view[idx].reshape(-1)
        y = self.tx_classes[idx + self.k]
        return x, y


def discover_symbol_files() -> Tuple[Path, List[int]]:
    for base_dir in Config.DATA_DIR_CANDIDATES:
        if not base_dir.exists():
            continue
        files = sorted(base_dir.glob("Symbols_1m_1ch_PR_*.csv"))
        if not files:
            continue
        indices = sorted([int(p.stem.split("_")[-1]) for p in files])[: Config.MAX_FILES]
        return base_dir, indices
    raise FileNotFoundError("No Symbols_1m_1ch_PR_*.csv files found.")


def load_files(base_dir: Path, file_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tx_list, rx_list = [], []
    for idx in file_indices:
        path = base_dir / f"Symbols_1m_1ch_PR_{idx}.csv"
        arr = pd.read_csv(path, header=None, dtype=np.float32).to_numpy(copy=False)
        tx_list.append(torch.from_numpy(arr[:, 0:2].copy()))
        rx_list.append(torch.from_numpy(arr[:, 2:4].copy()))
    return torch.cat(tx_list, dim=0), torch.cat(rx_list, dim=0)


def prepare_dataloaders():
    base_dir, all_indices = discover_symbol_files()
    train_files = max(1, int(len(all_indices) * Config.TRAIN_PORTION))
    if train_files >= len(all_indices):
        train_files = len(all_indices) - 1
    train_idx = all_indices[: train_files - 1]
    val_idx = [all_indices[train_files - 1]]
    test_idx = all_indices[train_files:]

    log(f"Data dir: {base_dir}")
    log(f"Train files: {train_idx}")
    log(f"Val files: {val_idx}")
    log(f"Test files: {test_idx}")

    tx_train, rx_train = load_files(base_dir, train_idx)
    tx_val, rx_val = load_files(base_dir, val_idx)
    tx_test, rx_test = load_files(base_dir, test_idx)

    tx_train_classes = symbols_to_classes(tx_train.to(Config.DEVICE)).cpu()
    tx_val_classes = symbols_to_classes(tx_val.to(Config.DEVICE)).cpu()
    tx_test_classes = symbols_to_classes(tx_test.to(Config.DEVICE)).cpu()

    train_ds = WindowedSymbolDataset(rx_train, tx_train_classes, Config.CONTEXT_K)
    val_ds = WindowedSymbolDataset(rx_val, tx_val_classes, Config.CONTEXT_K, mean=train_ds.mean, std=train_ds.std)
    test_ds = WindowedSymbolDataset(rx_test, tx_test_classes, Config.CONTEXT_K, mean=train_ds.mean, std=train_ds.std)

    loader_kwargs = {
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": Config.PIN_MEMORY,
        "persistent_workers": Config.NUM_WORKERS > 0,
    }
    if Config.NUM_WORKERS > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader, tx_test, rx_test


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, ce: Optional[nn.Module] = None) -> Tuple[float, float, float]:
    model.eval()
    if ce is None:
        ce = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    total_loss = 0.0
    total = 0
    correct = 0
    bit_errors = 0
    total_bits = 0
    bit_labels = BIT_LABELS.to(Config.DEVICE)

    for x, y in loader:
        x = x.to(Config.DEVICE, non_blocking=True)
        y = y.to(Config.DEVICE, non_blocking=True)
        for xb, yb, _ in iter_micro_batches(x, y):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP):
                mark_cudagraph_step_begin()
                logits = model(xb)
            loss = combined_ber_loss(logits, yb, ce)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * xb.size(0)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            tx_bits = bit_labels[yb]
            rx_bits = bit_labels[preds]
            bit_errors += (tx_bits != rx_bits).sum().item()
            total_bits += tx_bits.numel()

    return total_loss / max(total, 1), correct / max(total, 1), bit_errors / max(total_bits, 1)


def make_model(name: str) -> nn.Module:
    if name == "lstm":
        return LSTMRxEqualizer().to(Config.DEVICE)
    if name == "cnn_lstm":
        return HybridCNNLSTMEqualizer().to(Config.DEVICE)
    raise ValueError(f"Unknown model: {name}")


def train_one_model(model_name: str, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[nn.Module, Dict]:
    model = make_model(model_name)
    if Config.USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=Config.TORCH_COMPILE_MODE)
            log(f"{model_name} | torch.compile enabled ({Config.TORCH_COMPILE_MODE})")
        except Exception as compile_error:
            log(f"{model_name} | torch.compile disabled: {compile_error}")

    adamw_kwargs = dict(lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    if Config.DEVICE.type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = optim.AdamW(model.parameters(), **adamw_kwargs)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP)
    ema = EMAHelper(model, Config.EMA_DECAY) if Config.USE_EMA else None

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = Config.EPOCHS * steps_per_epoch
    warmup_steps = Config.WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_ber": [], "lr": [], "epoch_time_sec": []}
    best_val_ber = float("inf")
    steps_wo_min = 0
    total_seen = 0

    best_path = Config.OUT_DIR / f"{model_name}_best.pth"
    train_start = time.time()

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(Config.DEVICE, non_blocking=True)
            y = y.to(Config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for xb, yb, weight in iter_micro_batches(x, y):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP):
                    mark_cudagraph_step_begin()
                    logits = model(xb)
                loss = combined_ber_loss(logits, yb, criterion)
                scaler.scale(loss * weight).backward()
                batch_loss += loss.item() * xb.size(0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running_loss += batch_loss
            seen += x.size(0)
            total_seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        if ema is not None:
            ema.apply_shadow(model)
        val_loss, val_acc, val_ber = evaluate(model, val_loader, criterion)
        if ema is not None:
            ema.restore(model)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ber"].append(val_ber)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time_sec"].append(epoch_time)

        if epoch % Config.LOG_EVERY == 0:
            log(f"{model_name} | epoch {epoch+1} -- loss {train_loss:.6f} -- time {epoch_time:.3f}s")

        if val_ber < best_val_ber:
            best_val_ber = val_ber
            steps_wo_min = 0
            if Config.SAVE_BEST:
                if ema is not None:
                    ema.apply_shadow(model)
                torch.save({"model_state_dict": model.state_dict(), "history": history}, best_path)
                if ema is not None:
                    ema.restore(model)
                log(f"{model_name} | epoch {epoch+1} -- Model saved -- val_ber {val_ber:.6e}")
        else:
            steps_wo_min += 1

        if steps_wo_min >= Config.DECAY_STEPS and epoch >= Config.WARMUP_EPOCHS:
            log(f"{model_name} | Early stop by val_ber patience ({Config.DECAY_STEPS} epochs without BER improvement)")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    total_train_sec = time.time() - train_start
    history["total_train_sec"] = total_train_sec
    history["avg_epoch_sec"] = float(np.mean(history["epoch_time_sec"])) if history["epoch_time_sec"] else float("nan")
    history["samples_per_sec"] = total_seen / max(total_train_sec, 1e-8)
    return model, history


def plot_results(history: Dict, eval_results: Dict, model_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    train_epochs = list(range(1, len(history["train_loss"]) + 1))

    axes[0, 0].plot(train_epochs, history["train_loss"], label="Train", linewidth=2, alpha=0.8)
    axes[0, 0].plot(train_epochs, history["val_loss"], label="Val", linewidth=2)
    axes[0, 0].set_title(f"{model_name} - Loss Curves", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(train_epochs, history["val_acc"], color="green", linewidth=2)
    axes[0, 1].set_title(f"{model_name} - Validation Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    axes[0, 2].plot(train_epochs, history["lr"], color="red", linewidth=2)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("LR")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(
        ["Baseline", f"{model_name} EQ"],
        [eval_results["baseline_ber"], eval_results["equalized_ber"]],
        color=["#ff7675", "#55efc4"],
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("BER Comparison (log scale)", fontweight="bold")
    axes[1, 0].set_ylabel("BER")
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

    axes[1, 2].plot(train_epochs, [l * 100 for l in history["train_loss"]], label="Train Loss x100", alpha=0.7, linewidth=1)
    axes[1, 2].plot(train_epochs, [a * 100 for a in history["val_acc"]], label="Val Acc (%)", alpha=0.7, linewidth=2)
    axes[1, 2].set_title("Training Dynamics", fontweight="bold")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Value")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    plt.suptitle(f"{model_name} Equalizer Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_path = Config.OUT_DIR / f"ber_results_{model_name.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def final_eval(model: nn.Module, test_loader: DataLoader, tx_test: torch.Tensor, rx_test: torch.Tensor) -> Dict:
    model.eval()
    tx_cls = symbols_to_classes(tx_test.to(Config.DEVICE))
    rx_cls = symbols_to_classes(rx_test.to(Config.DEVICE))
    baseline_ber = calculate_ber_from_classes(tx_cls, rx_cls)

    bit_labels = BIT_LABELS.to(Config.DEVICE)
    total = 0
    correct = 0
    bit_errors = 0
    total_bits = 0
    for x, y in test_loader:
        x = x.to(Config.DEVICE, non_blocking=True)
        y = y.to(Config.DEVICE, non_blocking=True)
        for xb, yb, _ in iter_micro_batches(x, y):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=Config.DEVICE.type == "cuda" and Config.USE_AMP):
                mark_cudagraph_step_begin()
                logits = model(xb)
            pred = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
            tx_bits = bit_labels[yb]
            rx_bits = bit_labels[pred]
            bit_errors += (tx_bits != rx_bits).sum().item()
            total_bits += tx_bits.numel()

    equalized_ber = bit_errors / total_bits
    accuracy = correct / total
    ser = 1.0 - accuracy
    return {
        "baseline_ber": baseline_ber,
        "equalized_ber": equalized_ber,
        "ser": ser,
        "accuracy": accuracy,
        "improvement_abs": baseline_ber - equalized_ber,
        "improvement_rel": (1 - equalized_ber / baseline_ber) * 100,
        "improvement_db": 10 * np.log10(baseline_ber / equalized_ber) if equalized_ber > 0 else float("inf"),
    }


def main():
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Device: {Config.DEVICE}")
    train_loader, val_loader, test_loader, tx_test, rx_test = prepare_dataloaders()

    all_results = []
    for model_name in Config.MODELS:
        model, history = train_one_model(model_name, train_loader, val_loader)
        results = final_eval(model, test_loader, tx_test, rx_test)
        results["model_type"] = model_name
        results["train_total_sec"] = history.get("total_train_sec", float("nan"))
        results["avg_epoch_sec"] = history.get("avg_epoch_sec", float("nan"))
        results["samples_per_sec"] = history.get("samples_per_sec", float("nan"))
        results["mxnet_ref_ber_after"] = Config.MXNET_REF_BER_AFTER
        results["mxnet_ref_epoch_sec"] = Config.MXNET_REF_EPOCH_SEC
        results["ber_gain_vs_mxnet_x"] = Config.MXNET_REF_BER_AFTER / max(results["equalized_ber"], 1e-12)
        results["speedup_vs_mxnet_x"] = Config.MXNET_REF_EPOCH_SEC / max(results["avg_epoch_sec"], 1e-12)
        all_results.append(results)
        plot_results(history, results, model_name=model_name)
        torch.save(model.state_dict(), Config.OUT_DIR / f"{model_name}_final.pth")
        log(
            f"{model_name.upper()} | BER {results['equalized_ber']:.6e} | "
            f"Acc {results['accuracy']:.4%} | RelImp {results['improvement_rel']:.2f}% | "
            f"Epoch {results['avg_epoch_sec']:.2f}s | "
            f"BERxMX {results['ber_gain_vs_mxnet_x']:.2f} | SpeedxMX {results['speedup_vs_mxnet_x']:.2f}"
        )
        if Config.DEVICE.type == "cuda":
            model.to("cpu")
            del model
            torch.cuda.empty_cache()

    summary = pd.DataFrame(all_results).sort_values("equalized_ber").reset_index(drop=True)
    summary_path = Config.OUT_DIR / "architecture_comparison.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Saved comparison: {summary_path}")


if __name__ == "__main__":
    main()
