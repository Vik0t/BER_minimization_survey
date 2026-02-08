import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time
from models import LSTMRxEqualizer, HybridCNN_LSTM_Equalizer
import warnings
warnings.filterwarnings("ignore")

class InferenceEngine:
    """Движок для выполнения инференса"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.constellation = torch.tensor([
            [-0.948683, -0.948683], [-0.948683, -0.316228], [-0.948683,  0.316228], [-0.948683,  0.948683],
            [-0.316228, -0.948683], [-0.316228, -0.316228], [-0.316228,  0.316228], [-0.316228,  0.948683],
            [ 0.316228, -0.948683], [ 0.316228, -0.316228], [ 0.316228,  0.316228], [ 0.316228,  0.948683],
            [ 0.948683, -0.948683], [ 0.948683, -0.316228], [ 0.948683,  0.316228], [ 0.948683,  0.948683],
        ], device="cpu")
        
        self.bit_labels = torch.tensor([
            [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,0,1,0],
            [0,1,0,0], [0,1,0,1], [0,1,1,1], [0,1,1,0],
            [1,1,0,0], [1,1,0,1], [1,1,1,1], [1,1,1,0],
            [1,0,0,0], [1,0,0,1], [1,0,1,1], [1,0,1,0]
        ], device="cpu", dtype=torch.uint8)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Inference engine initialized on {self.device}")
    
    def load_models_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию моделей"""
        config_file = self.models_dir / 'models_config.json'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            'models': [
                {
                    'id': 'lstm',
                    'name': 'LSTM Equalizer',
                    'description': 'Bidirectional LSTM with attention mechanism',
                    'file': 'lstm_model.pth',
                    'type': 'lstm',
                    'parameters': '696K',
                    'accuracy': '97.7%',
                    'ber_improvement': '44.1%'
                },
                {
                    'id': 'cnn_lstm',
                    'name': 'CNN-LSTM Hybrid',
                    'description': 'CNN feature extractor + LSTM temporal modeling',
                    'file': 'cnn_lstm_model.pth',
                    'type': 'hybrid',
                    'parameters': '1.2M',
                    'accuracy': '98.2%',
                    'ber_improvement': '52.3%'
                },
                {
                    'id': 'light_cnn',
                    'name': 'Light CNN',
                    'description': 'Lightweight CNN for fast inference',
                    'file': 'cnn_light.pth',
                    'type': 'cnn',
                    'parameters': '256K',
                    'accuracy': '96.5%',
                    'ber_improvement': '38.7%'
                }
            ]
        }
        
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def load_model(self, model_id: str) -> torch.nn.Module:
        """Загружает модель по ID"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        # Находим модель в конфиге
        config = self.load_models_config()
        model_info = next((m for m in config['models'] if m['id'] == model_id), None)
        
        if not model_info:
            raise ValueError(f"Model {model_id} not found in configuration")
        
        model_path = self.models_dir / model_info['file']
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config_dict = checkpoint.get('config', {})
        
        # Создаем модель
        if model_info['type'] == 'lstm':
            model = LSTMRxEqualizer(input_dim=2, config=config_dict)
        elif model_info['type'] == 'hybrid':
            model = HybridCNN_LSTM_Equalizer(input_dim=2, config=config_dict)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Кэшируем модель
        self.loaded_models[model_id] = {
            'model': model,
            'norm_mean': checkpoint.get('normalization_mean'),
            'norm_std': checkpoint.get('normalization_std'),
            'context_k': checkpoint.get('context_k', 40),
            'info': model_info
        }
        
        print(f"✓ Model {model_id} loaded successfully")
        return self.loaded_models[model_id]
    
    def preprocess_data(self, filepath: str) -> Dict[str, Any]:
        """Предобрабатывает входные данные"""
        df = pd.read_csv(filepath, header=None)
        
        has_tx = df.shape[1] >= 4
        
        if has_tx:
            tx_symbols = df.iloc[:, 0:2].values
            rx_symbols = df.iloc[:, 2:4].values
        else:
            tx_symbols = None
            rx_symbols = df.iloc[:, :2].values
        
        return {
            'tx_symbols': tx_symbols,
            'rx_symbols': rx_symbols,
            'has_tx': has_tx,
            'num_symbols': len(rx_symbols)
        }
    
    def create_windows(self, rx_symbols: np.ndarray, context_k: int, 
                      mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Создает окна для инференса"""
        windows = []
        
        for i in range(context_k, len(rx_symbols) - context_k):
            window = rx_symbols[i - context_k:i + context_k + 1].reshape(-1)
            windows.append(window)
        
        windows_tensor = torch.tensor(windows, dtype=torch.float32)
        
        # Нормализация
        if mean is not None and std is not None:
            windows_tensor = (windows_tensor - mean) / std
        
        return windows_tensor
    
    def predict(self, model: torch.nn.Module, windows: torch.Tensor, 
               batch_size: int = 256) -> Dict[str, Any]:
        """Выполняет предсказания"""
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size].to(self.device)
                
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(batch)
                else:
                    outputs = model(batch)
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())
        
        pred_classes = torch.cat(all_predictions)
        pred_symbols = self.constellation[pred_classes].numpy()
        pred_bits = self.bit_labels[pred_classes].numpy()
        
        return {
            'classes': pred_classes.numpy().tolist(),
            'symbols': pred_symbols.tolist(),
            'bits': pred_bits.tolist(),
            'num_predictions': len(pred_classes)
        }
    
    def calculate_metrics(self, pred_bits: np.ndarray, tx_bits: Optional[np.ndarray] = None,
                         rx_bits: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Вычисляет метрики качества"""
        if tx_bits is None:
            return {}
        
        # Вычисляем BER
        pred_bits_tensor = torch.tensor(pred_bits)
        tx_bits_tensor = torch.tensor(tx_bits)
        rx_bits_tensor = torch.tensor(rx_bits) if rx_bits is not None else None
        
        # Equalized BER
        equalized_ber = (pred_bits_tensor != tx_bits_tensor).float().mean().item()
        
        # Baseline BER (если есть RX биты)
        if rx_bits_tensor is not None:
            baseline_ber = (rx_bits_tensor != tx_bits_tensor).float().mean().item()
        else:
            baseline_ber = None
        
        # Вычисляем улучшение
        if baseline_ber is not None and baseline_ber > 0:
            improvement_abs = baseline_ber - equalized_ber
            improvement_rel = (1 - equalized_ber / baseline_ber) * 100
            improvement_db = 10 * np.log10(baseline_ber / equalized_ber) if equalized_ber > 0 else float('inf')
        else:
            improvement_abs = improvement_rel = improvement_db = None
        
        # Accuracy
        accuracy = (pred_bits_tensor == tx_bits_tensor).all(dim=1).float().mean().item()
        
        return {
            'baseline_ber': baseline_ber,
            'equalized_ber': equalized_ber,
            'accuracy': accuracy,
            'improvement_abs': improvement_abs,
            'improvement_rel': improvement_rel,
            'improvement_db': improvement_db,
            'symbol_error_rate': 1 - accuracy
        }
    
    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Преобразует символы в биты"""
        symbols_tensor = torch.tensor(symbols)
        diff = symbols_tensor.unsqueeze(1) - self.constellation.unsqueeze(0)
        dist = torch.sum(diff ** 2, dim=2)
        classes = torch.argmin(dist, dim=1)
        return self.bit_labels[classes].numpy()
    
    def run_inference(self, model_id: str, input_file: str, 
                     session_id: str, result_id: str) -> Dict[str, Any]:
        """Основная функция инференса"""
        start_time = time.time()
        
        try:
            # 1. Загружаем модель
            model_data = self.load_model(model_id)
            model = model_data['model']
            
            # 2. Загружаем данные
            data = self.preprocess_data(input_file)
            
            # 3. Создаем окна
            windows = self.create_windows(
                data['rx_symbols'],
                model_data['context_k'],
                model_data['norm_mean'],
                model_data['norm_std']
            )
            
            # 4. Выполняем предсказания
            predictions = self.predict(model, windows)
            
            # 5. Вычисляем метрики (если есть TX символы)
            metrics = {}
            if data['has_tx'] and data['tx_symbols'] is not None:
                # Берем центральные символы
                context_k = model_data['context_k']
                tx_center = data['tx_symbols'][context_k:-context_k]
                rx_center = data['rx_symbols'][context_k:-context_k]
                
                # Преобразуем в биты
                tx_bits = self.symbols_to_bits(tx_center)
                rx_bits = self.symbols_to_bits(rx_center)
                
                # Вычисляем метрики
                metrics = self.calculate_metrics(
                    predictions['bits'],
                    tx_bits,
                    rx_bits
                )
            
            # 6. Формируем результат
            result = {
                'success': True,
                'result_id': result_id,
                'session_id': session_id,
                'model_id': model_id,
                'model_info': model_data['info'],
                'data': {
                    'num_symbols': data['num_symbols'],
                    'has_tx': data['has_tx'],
                    'rx_symbols': {
                        'I': data['rx_symbols'][:, 0].tolist(),
                        'Q': data['rx_symbols'][:, 1].tolist()
                    }
                },
                'predictions': predictions,
                'metrics': metrics,
                'processing_time': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"✓ Inference completed in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'result_id': result_id,
                'session_id': session_id,
                'processing_time': time.time() - start_time
            }