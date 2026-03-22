import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from models import build_model

warnings.filterwarnings("ignore")


class InferenceEngine:
    """Движок для выполнения инференса"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.constellation = torch.tensor(
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
            device="cpu",
            dtype=torch.float32,
        )

        self.bit_labels = torch.tensor(
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
            device="cpu",
            dtype=torch.uint8,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Inference engine initialized on {self.device}")

    def load_models_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию моделей"""
        config_file = self.models_dir / "models_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"models": []}

    def _extract_runtime_config(self, checkpoint: Any, model_info: Dict[str, Any]) -> Dict[str, Any]:
        config_dict: Dict[str, Any] = {}
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            config_dict.update(checkpoint["config"])

        config_dict.update(model_info.get("runtime_config", {}))
        config_dict.setdefault("CONTEXT_K", model_info.get("context_k", 40))
        return config_dict

    def _extract_state_dict(self, checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict):
            tensor_values = [value for value in checkpoint.values() if isinstance(value, torch.Tensor)]
            if tensor_values and len(tensor_values) == len(checkpoint):
                return checkpoint
        raise ValueError("Unsupported checkpoint format: expected state_dict or checkpoint with model_state_dict")

    def _extract_norm_stats(
        self,
        checkpoint: Any,
        model_info: Dict[str, Any],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mean = None
        std = None

        if isinstance(checkpoint, dict):
            mean = checkpoint.get("normalization_mean")
            std = checkpoint.get("normalization_std")

        if mean is None:
            mean = model_info.get("normalization_mean")
        if std is None:
            std = model_info.get("normalization_std")

        if mean is not None:
            mean = torch.as_tensor(mean, dtype=torch.float32)
        if std is not None:
            std = torch.as_tensor(std, dtype=torch.float32)
            std = torch.where(std == 0, torch.ones_like(std), std)

        return mean, std

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Загружает модель по ID"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        config = self.load_models_config()
        model_info = next((m for m in config["models"] if m["id"] == model_id), None)
        if not model_info:
            raise ValueError(f"Model {model_id} not found in configuration")

        model_path = self.models_dir / model_info["file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        runtime_config = self._extract_runtime_config(checkpoint, model_info)
        state_dict = self._extract_state_dict(checkpoint)
        model = build_model(model_info["type"], input_dim=2, config=runtime_config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        norm_mean, norm_std = self._extract_norm_stats(checkpoint, model_info)
        context_k = int(runtime_config.get("CONTEXT_K", model_info.get("context_k", 40)))

        self.loaded_models[model_id] = {
            "model": model,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "context_k": context_k,
            "info": model_info,
            "output_mode": model_info.get("output_mode", "auto"),
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
            "tx_symbols": tx_symbols,
            "rx_symbols": rx_symbols,
            "has_tx": has_tx,
            "num_symbols": len(rx_symbols),
        }

    def create_windows(
        self,
        rx_symbols: np.ndarray,
        context_k: int,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Создает окна для инференса"""
        rx_tensor = torch.as_tensor(rx_symbols, dtype=torch.float32)

        if mean is None or std is None:
            mean = rx_tensor.mean(dim=0, keepdim=True)
            std = rx_tensor.std(dim=0, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
        else:
            mean = mean.reshape(1, -1).to(dtype=torch.float32)
            std = std.reshape(1, -1).to(dtype=torch.float32)
            if mean.numel() > 2:
                mean = mean.reshape(-1, 2).mean(dim=0, keepdim=True)
            if std.numel() > 2:
                std = std.reshape(-1, 2).mean(dim=0, keepdim=True)

        norm_rx = (rx_tensor - mean) / std
        norm_rx = torch.nan_to_num(norm_rx, nan=0.0, posinf=0.0, neginf=0.0)
        window_view = norm_rx.unfold(0, 2 * context_k + 1, 1).permute(0, 2, 1).contiguous()
        return window_view.view(window_view.size(0), -1).contiguous()

    def _outputs_to_classes(self, outputs: torch.Tensor, output_mode: str = "auto") -> torch.Tensor:
        if outputs.ndim != 2:
            raise ValueError(f"Unexpected output shape: {tuple(outputs.shape)}")

        if output_mode == "class_logits" or outputs.shape[1] == 16:
            return torch.argmax(outputs, dim=1)
        if output_mode in {"symbol_regression", "auto"} and outputs.shape[1] == 2:
            diff = outputs.unsqueeze(1).cpu() - self.constellation.unsqueeze(0)
            dist = torch.sum(diff ** 2, dim=2)
            return torch.argmin(dist, dim=1)

        raise ValueError(f"Unsupported output dimension {outputs.shape[1]} for mode {output_mode}")

    def predict(
        self,
        model: torch.nn.Module,
        windows: torch.Tensor,
        batch_size: int = 256,
        output_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Выполняет предсказания"""
        model.eval()
        all_classes = []

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i : i + batch_size].to(self.device)
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(batch)
                else:
                    outputs = model(batch)
                pred_classes = self._outputs_to_classes(outputs.detach(), output_mode=output_mode)
                all_classes.append(pred_classes.cpu())

        pred_classes = torch.cat(all_classes)
        pred_symbols = self.constellation[pred_classes].numpy()
        pred_bits = self.bit_labels[pred_classes].numpy()

        return {
            "classes": pred_classes.numpy().tolist(),
            "symbols": pred_symbols.tolist(),
            "bits": pred_bits.tolist(),
            "num_predictions": len(pred_classes),
        }

    def calculate_metrics(
        self,
        pred_bits: np.ndarray,
        tx_bits: Optional[np.ndarray] = None,
        rx_bits: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Вычисляет метрики качества"""
        if tx_bits is None:
            return {}

        pred_bits_tensor = torch.tensor(pred_bits)
        tx_bits_tensor = torch.tensor(tx_bits)
        rx_bits_tensor = torch.tensor(rx_bits) if rx_bits is not None else None

        equalized_ber = (pred_bits_tensor != tx_bits_tensor).float().mean().item()

        if rx_bits_tensor is not None:
            baseline_ber = (rx_bits_tensor != tx_bits_tensor).float().mean().item()
        else:
            baseline_ber = None

        if baseline_ber is not None and baseline_ber > 0:
            improvement_abs = baseline_ber - equalized_ber
            improvement_rel = (1 - equalized_ber / baseline_ber) * 100
            improvement_db = 10 * np.log10(baseline_ber / equalized_ber) if equalized_ber > 0 else float("inf")
        else:
            improvement_abs = improvement_rel = improvement_db = None

        accuracy = (pred_bits_tensor == tx_bits_tensor).all(dim=1).float().mean().item()

        return {
            "baseline_ber": baseline_ber,
            "equalized_ber": equalized_ber,
            "accuracy": accuracy,
            "improvement_abs": improvement_abs,
            "improvement_rel": improvement_rel,
            "improvement_db": improvement_db,
            "symbol_error_rate": 1 - accuracy,
        }

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Преобразует символы в биты"""
        symbols_tensor = torch.tensor(symbols, dtype=torch.float32)
        diff = symbols_tensor.unsqueeze(1) - self.constellation.unsqueeze(0)
        dist = torch.sum(diff ** 2, dim=2)
        classes = torch.argmin(dist, dim=1)
        return self.bit_labels[classes].numpy()

    def run_inference(
        self,
        model_id: str,
        input_file: str,
        session_id: str,
        result_id: str,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """Основная функция инференса"""
        start_time = time.time()

        try:
            model_data = self.load_model(model_id)
            model = model_data["model"]
            data = self.preprocess_data(input_file)

            windows = self.create_windows(
                data["rx_symbols"],
                model_data["context_k"],
                model_data["norm_mean"],
                model_data["norm_std"],
            )

            predictions = self.predict(
                model,
                windows,
                batch_size=batch_size,
                output_mode=model_data["output_mode"],
            )

            metrics = {}
            if data["has_tx"] and data["tx_symbols"] is not None:
                context_k = model_data["context_k"]
                tx_center = data["tx_symbols"][context_k:-context_k]
                rx_center = data["rx_symbols"][context_k:-context_k]
                tx_bits = self.symbols_to_bits(tx_center)
                rx_bits = self.symbols_to_bits(rx_center)
                metrics = self.calculate_metrics(predictions["bits"], tx_bits, rx_bits)

            result = {
                "success": True,
                "result_id": result_id,
                "session_id": session_id,
                "model_id": model_id,
                "model_info": model_data["info"],
                "data": {
                    "num_symbols": data["num_symbols"],
                    "has_tx": data["has_tx"],
                    "rx_symbols": {
                        "I": data["rx_symbols"][:, 0].tolist(),
                        "Q": data["rx_symbols"][:, 1].tolist(),
                    },
                },
                "predictions": predictions,
                "metrics": metrics,
                "processing_time": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            print(f"✓ Inference completed in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "result_id": result_id,
                "session_id": session_id,
                "processing_time": time.time() - start_time,
            }
