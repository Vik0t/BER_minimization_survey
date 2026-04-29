import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# КОНСТАНТЫ (должны совпадать с обучением)
# ============================================================================
CONSTELLATION = torch.tensor([
    [-0.948683, -0.948683], [-0.948683, -0.316228], [-0.948683,  0.316228], [-0.948683,  0.948683],
    [-0.316228, -0.948683], [-0.316228, -0.316228], [-0.316228,  0.316228], [-0.316228,  0.948683],
    [ 0.316228, -0.948683], [ 0.316228, -0.316228], [ 0.316228,  0.316228], [ 0.316228,  0.948683],
    [ 0.948683, -0.948683], [ 0.948683, -0.316228], [ 0.948683,  0.316228], [ 0.948683,  0.948683],
], device="cpu")

BIT_LABELS = torch.tensor([
    [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,0,1,0],
    [0,1,0,0], [0,1,0,1], [0,1,1,1], [0,1,1,0],
    [1,1,0,0], [1,1,0,1], [1,1,1,1], [1,1,1,0],
    [1,0,0,0], [1,0,0,1], [1,0,1,1], [1,0,1,0]
], device="cpu", dtype=torch.uint8)

# ============================================================================
# АРХИТЕКТУРА МОДЕЛИ (должна совпадать с обученной)
# ============================================================================
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
    def __init__(self, input_dim: int = 2, config: dict = None):
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
        self.use_attention = self.use_attention
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

# ============================================================================
# ФУНКЦИИ ДЛЯ ИНФЕРЕНСА
# ============================================================================
def load_model(model_path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    """Загружает сохраненную модель"""
    print(f"Loading model from {model_path}...")
    
    # Загружаем checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Извлекаем конфигурацию
    config = checkpoint.get('config', {})
    
    # Создаем модель с правильной архитектурой
    model_class_name = checkpoint.get('model_architecture', 'LSTMRxEqualizer')
    
    if model_class_name == 'LSTMRxEqualizer':
        model = LSTMRxEqualizer(input_dim=checkpoint['input_dim'], config=config)
    else:
        raise ValueError(f"Unknown model architecture: {model_class_name}")
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Architecture: {model_class_name}")
    print(f"  Context window: ±{config.get('CONTEXT_K', 'unknown')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Возвращаем модель и информацию о нормализации
    norm_info = {
        'mean': checkpoint.get('normalization_mean'),
        'std': checkpoint.get('normalization_std'),
        'context_k': checkpoint.get('context_k', 40)
    }
    
    return model, norm_info

def load_csv_file(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Загружает CSV файл с символами"""
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, header=None)
        
        # Предполагаем формат: tx_i, tx_q, rx_i, rx_q
        if df.shape[1] >= 4:
            tx_symbols = torch.tensor(df.iloc[:, 0:2].values, dtype=torch.float32)
            rx_symbols = torch.tensor(df.iloc[:, 2:4].values, dtype=torch.float32)
            print(f"✓ Loaded {len(tx_symbols):,} symbols")
            print(f"  TX shape: {tx_symbols.shape}, RX shape: {rx_symbols.shape}")
            return tx_symbols, rx_symbols
        else:
            # Если только RX символы
            rx_symbols = torch.tensor(df.iloc[:, :2].values, dtype=torch.float32)
            print(f"✓ Loaded {len(rx_symbols):,} RX symbols only")
            return None, rx_symbols
            
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None, None

def create_windows(rx_symbols: torch.Tensor, context_k: int, 
                   mean: torch.Tensor = None, std: torch.Tensor = None) -> torch.Tensor:
    """Создает окна для инференса"""
    windows = []
    
    for i in range(context_k, len(rx_symbols) - context_k):
        window = rx_symbols[i - context_k:i + context_k + 1].reshape(-1)
        windows.append(window)
    
    windows_tensor = torch.stack(windows)
    
    # Нормализация, если есть параметры
    if mean is not None and std is not None:
        windows_tensor = (windows_tensor - mean) / std
    
    return windows_tensor

def predict_symbols(model: nn.Module, windows: torch.Tensor, 
                   batch_size: int = 256, device: str = "cpu") -> torch.Tensor:
    """Делает предсказания для всех окон"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        # Разбиваем на батчи
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size].to(device)
            
            # Forward pass
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
            else:
                outputs = model(batch)
            
            # Получаем предсказанные классы
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
    
    return torch.cat(all_predictions)

def classes_to_symbols(classes: torch.Tensor) -> torch.Tensor:
    """Преобразует классы в символы (I, Q)"""
    return CONSTELLATION[classes]

def classes_to_bits(classes: torch.Tensor) -> torch.Tensor:
    """Преобразует классы в биты"""
    return BIT_LABELS[classes]

def calculate_ber(pred_bits: torch.Tensor, true_bits: torch.Tensor) -> float:
    """Вычисляет BER между предсказанными и истинными битами"""
    if true_bits is None:
        return None
    
    errors = (pred_bits != true_bits).float().sum()
    total_bits = pred_bits.numel()
    return (errors / total_bits).item()

def visualize_results(rx_symbols: torch.Tensor, pred_symbols: torch.Tensor, 
                     tx_symbols: torch.Tensor = None, save_path: str = None):
    """Визуализирует результаты инференса"""
    fig, axes = plt.subplots(1, 3 if tx_symbols is not None else 2, figsize=(15, 5))
    
    # 1. Принятые символы
    axes[0].scatter(rx_symbols[:, 0].numpy(), rx_symbols[:, 1].numpy(), 
                   alpha=0.3, s=5, label='RX symbols', color='blue')
    axes[0].scatter(CONSTELLATION[:, 0], CONSTELLATION[:, 1], 
                   marker='x', s=100, color='red', label='Constellation')
    axes[0].set_title('Received Symbols', fontweight='bold')
    axes[0].set_xlabel('I'); axes[0].set_ylabel('Q')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].axis('equal')
    
    # 2. Выровненные символы
    axes[1].scatter(pred_symbols[:, 0].numpy(), pred_symbols[:, 1].numpy(), 
                   alpha=0.3, s=5, label='Equalized', color='green')
    axes[1].scatter(CONSTELLATION[:, 0], CONSTELLATION[:, 1], 
                   marker='x', s=100, color='red', label='Constellation')
    axes[1].set_title('Equalized Symbols', fontweight='bold')
    axes[1].set_xlabel('I'); axes[1].set_ylabel('Q')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].axis('equal')
    
    # 3. Переданные символы (если есть)
    if tx_symbols is not None:
        axes[2].scatter(tx_symbols[:, 0].numpy(), tx_symbols[:, 1].numpy(), 
                       alpha=0.3, s=5, label='TX symbols', color='purple')
        axes[2].scatter(CONSTELLATION[:, 0], CONSTELLATION[:, 1], 
                       marker='x', s=100, color='red', label='Constellation')
        axes[2].set_title('Transmitted Symbols', fontweight='bold')
        axes[2].set_xlabel('I'); axes[2].set_ylabel('Q')
        axes[2].legend(); axes[2].grid(alpha=0.3)
        axes[2].axis('equal')
    
    plt.suptitle('Symbol Equalization Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    
    plt.show()

# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ИНФЕРЕНСА
# ============================================================================
def run_inference(model_path: str, data_file: str, output_dir: str = "inference_results",
                  batch_size: int = 256, device: str = None):
    """
    Основная функция для запуска инференса
    
    Args:
        model_path: путь к сохраненной модели
        data_file: путь к CSV файлу с данными
        output_dir: директория для сохранения результатов
        batch_size: размер батча для инференса
        device: устройство для вычислений (cuda/cpu)
    """
    # Определяем устройство
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("SYMBOL EQUALIZER INFERENCE")
    print("=" * 80)
    
    # 1. Загружаем модель
    model, norm_info = load_model(model_path, device=device)
    
    # 2. Загружаем данные
    tx_symbols, rx_symbols = load_csv_file(data_file)
    if rx_symbols is None:
        print("✗ Failed to load data")
        return
    
    # 3. Создаем окна
    context_k = norm_info['context_k']
    windows = create_windows(rx_symbols, context_k, norm_info['mean'], norm_info['std'])
    print(f"✓ Created {len(windows):,} windows for inference")
    
    # 4. Делаем предсказания
    print("\nRunning inference...")
    pred_classes = predict_symbols(model, windows, batch_size, device=device)
    
    # 5. Преобразуем в символы и биты
    pred_symbols = classes_to_symbols(pred_classes)
    pred_bits = classes_to_bits(pred_classes)
    
    print(f"✓ Predicted {len(pred_symbols):,} symbols")
    
    # 6. Вычисляем метрики (если есть TX символы)
    if tx_symbols is not None:
        # Берем центральные символы (те, для которых делали предсказания)
        tx_center = tx_symbols[context_k:len(tx_symbols) - context_k]
        
        # Преобразуем TX символы в классы
        diff = tx_center.unsqueeze(1) - CONSTELLATION.unsqueeze(0)
        dist = torch.sum(diff ** 2, dim=2)
        tx_classes = torch.argmin(dist, dim=1)
        tx_bits = BIT_LABELS[tx_classes]
        
        # Вычисляем BER
        ber = calculate_ber(pred_bits, tx_bits)
        
        # Baseline BER (без выравнивания)
        rx_classes = torch.argmin(
            torch.sum((rx_symbols[context_k:len(rx_symbols) - context_k].unsqueeze(1) - CONSTELLATION.unsqueeze(0)) ** 2, dim=2), 
            dim=1
        )
        rx_bits = BIT_LABELS[rx_classes]
        baseline_ber = calculate_ber(rx_bits, tx_bits)
        
        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS:")
        print("=" * 80)
        print(f"Baseline BER (no equalization): {baseline_ber:.6e}")
        print(f"Equalized BER:                  {ber:.6e}")
        print(f"Improvement:                    {((baseline_ber - ber) / baseline_ber * 100):.1f}%")
        print(f"SNR Gain:                       {10 * np.log10(baseline_ber / ber) if ber > 0 else float('inf'):.2f} dB")
    
    # 7. Создаем директорию для результатов
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 8. Сохраняем результаты
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем предсказанные символы
    symbols_df = pd.DataFrame(pred_symbols.numpy(), columns=['I_pred', 'Q_pred'])
    symbols_file = output_path / f"predicted_symbols_{timestamp}.csv"
    symbols_df.to_csv(symbols_file, index=False)
    print(f"✓ Predicted symbols saved to: {symbols_file}")
    
    # Сохраняем предсказанные биты
    bits_df = pd.DataFrame(pred_bits.numpy(), columns=['b0', 'b1', 'b2', 'b3'])
    bits_file = output_path / f"predicted_bits_{timestamp}.csv"
    bits_df.to_csv(bits_file, index=False)
    print(f"✓ Predicted bits saved to: {bits_file}")
    
    # 9. Визуализируем результаты
    vis_file = output_path / f"visualization_{timestamp}.png"
    visualize_results(
        rx_symbols[context_k:len(rx_symbols) - context_k],
        pred_symbols,
        tx_center if tx_symbols is not None else None,
        save_path=str(vis_file)
    )
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 80)

# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================
if __name__ == "__main__":
    # Пример 1: Инференс на тестовом файле с оценкой качества
    run_inference(
        model_path="ber_equalizer_lstm_final.pth",  # Ваша сохраненная модель
        data_file="Symbols_1m_1ch_PR_10.csv",       # Файл с данными (с TX и RX)
        output_dir="inference_results",
        batch_size=256,
        device="cuda"  # или "cpu"
    )
    
    # Пример 2: Инференс только на RX символах (без TX для сравнения)
    # run_inference(
    #     model_path="ber_equalizer_lstm_final.pth",
    #     data_file="only_rx_symbols.csv",  # Файл только с RX символами
    #     output_dir="inference_results",
    #     batch_size=256
    # )
    
    # Пример 3: Пакетная обработка нескольких файлов
    # for file_idx in [11, 12, 13]:
    #     print(f"\nProcessing file {file_idx}...")
    #     run_inference(
    #         model_path="ber_equalizer_lstm_final.pth",
    #         data_file=f"Symbols_1m_1ch_PR_{file_idx}.csv",
    #         output_dir=f"inference_results/file_{file_idx}",
    #         batch_size=256
    #     )