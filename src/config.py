from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Model parameters
    MODEL_NAME: str = "lstm"  # "mlp" or "cnn"

    context_k: int = 30  # Context window size
    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 3e-4
    DROPOUT: float = 0.2
    HIDDEN_DIM: int = 512
    EPOCHS: int = 1000
    PATIENCE: int = 60
    GRAD_CLIP: float = 0.3
    
    # Augmentation and loss
    USE_NOISE_AUG: bool = True
    NOISE_STD: float = 0.008  # Noise augmentation standard deviation
    USE_FOCAL_LOSS: bool = True  # Use focal loss
    GAMMA: float = 2.0  # Focal loss gamma parameter
    
    # Data paths
    DATA_DIR: Path = Path("Symbols_1m_1ch_PR")
    USE_CNN: bool = True
    
    # File indices for training and testing
    train_files = list(range(1, 10))  # Files 1-9 for training/validation
    test_files = [10]  # File 10 for testing
    
    # Compatibility properties
    @property
    def CONTEXT_K(self):
        return self.context_k
    
    @property
    def batch_size(self):
        return self.BATCH_SIZE
    
    @property
    def lr(self):
        return self.LEARNING_RATE
    
    @property
    def weight_decay(self):
        return self.WEIGHT_DECAY
    
    @property
    def dropout(self):
        return self.DROPOUT
    
    @property
    def hidden_dim(self):
        return self.HIDDEN_DIM
    
    @property
    def epochs(self):
        return self.EPOCHS
    
    @property
    def patience(self):
        return self.PATIENCE
    
    @property
    def grad_clip(self):
        return self.GRAD_CLIP
    
    @property
    def use_focal(self):
        return self.USE_FOCAL_LOSS
    
    @property
    def gamma(self):
        return self.GAMMA
    
    @property
    def model_name(self):
        return self.MODEL_NAME

# Initialize configuration
CFG = Config()

# CUDA optimizations
if CFG.DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
