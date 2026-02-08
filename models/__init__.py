from .mlp import MLP
from .cnn import EnhancedCNNRxEqualizer
from .lstm import LSTM_EQ

def build_model(name):
    if name == "mlp": return MLP()
    if name == "cnn": return EnhancedCNNRxEqualizer()
    if name == "lstm": return LSTM_EQ()
    raise ValueError("Unknown model")
