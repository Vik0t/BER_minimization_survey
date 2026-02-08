import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from src.config import CFG

class WindowedSymbolDataset(Dataset):
    def __init__(self, rx_symbols: torch.Tensor, tx_classes: torch.Tensor,
                 k: int = 30, normalize: bool = True, noise_std: float = 0.0):
        self.k = k if k is not None else CFG.context_k
        self.noise_std = noise_std
        self.X, self.y = self._create_windows(rx_symbols, tx_classes)

        if normalize:
            self.mean = self.X.mean(dim=0, keepdim=True)
            self.std = self.X.std(dim=0, keepdim=True)
            self.std[self.std == 0] = 1.0
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        self.training = False

    def _create_windows(self, rx_symbols: torch.Tensor, tx_classes: torch.Tensor):
        X, y = [], []
        for i in range(self.k, len(rx_symbols) - self.k):
            window = rx_symbols[i - self.k:i + self.k + 1].reshape(-1)
            X.append(window)
            y.append(tx_classes[i])
        return torch.stack(X), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.noise_std > 0 and self.training:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y

    def set_training(self, is_training: bool):
        self.training = is_training


def load_all_files(file_indices: list) -> tuple:
    """Загружает ВСЕ указанные файлы без пропусков."""
    tx_list, rx_list = [], []
    base_path = CFG.DATA_DIR if CFG.DATA_DIR.exists() else Path(".")

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
            raise FileNotFoundError(f"File for index {idx} not found in any expected location")

        df = pd.read_csv(file_path, header=None)
        tx_symbols = torch.tensor(df.iloc[:, 0:2].values, dtype=torch.float32)
        rx_symbols = torch.tensor(df.iloc[:, 2:4].values, dtype=torch.float32)
        tx_list.append(tx_symbols)
        rx_list.append(rx_symbols)
        print(f"  ✓ File {idx}: {len(tx_symbols):,} symbols")

    tx_all = torch.cat(tx_list, dim=0)
    rx_all = torch.cat(rx_list, dim=0)
    print(f"  → TOTAL: {len(tx_all):,} symbols from {len(file_indices)} files\n")
    return tx_all, rx_all
