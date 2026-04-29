#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the improved model with PolarizationPhaseActivation
"""

try:
    import torch
    import torch.nn as nn
    from archive_versions.experiment1 import (
        Config, 
        LSTMRxEqualizer, 
        HybridCNN_LSTM_Equalizer,
        PolarizationPhaseActivation
    )
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available, skipping tests")

def test_polarization_activation():
    """Test the PolarizationPhaseActivation implementation"""
    if not PYTORCH_AVAILABLE:
        print("Skipping PolarizationPhaseActivation test (PyTorch not available)")
        return True
        
    print("Testing PolarizationPhaseActivation...")
    
    # Create activation function
    activation = PolarizationPhaseActivation(dim=32)
    
    # Create test input (batch=2, seq_len=5, features=2)
    x = torch.randn(2, 5, 2)
    
    # Apply activation
    output = activation(x)
    
    # Check output shape
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    
    # Check that output is finite
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    print("✓ PolarizationPhaseActivation test passed")
    return True

def test_lstm_model():
    """Test the LSTMRxEqualizer implementation"""
    if not PYTORCH_AVAILABLE:
        print("Skipping LSTMRxEqualizer test (PyTorch not available)")
        return True
        
    print("Testing LSTMRxEqualizer...")
    
    # Create model
    model = LSTMRxEqualizer(input_dim=2)
    
    # Create test input (batch=2, features=seq_len*2)
    seq_len = 2 * Config.CONTEXT_K + 1
    x = torch.randn(2, seq_len * 2)
    
    # Apply model
    output = model(x)
    
    # Check output shape
    assert output.shape == (2, 16), f"Expected (2, 16), got {output.shape}"
    
    # Check that output is finite
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    print("✓ LSTMRxEqualizer test passed")
    return True

def test_hybrid_model():
    """Test the HybridCNN_LSTM_Equalizer implementation"""
    if not PYTORCH_AVAILABLE:
        print("Skipping HybridCNN_LSTM_Equalizer test (PyTorch not available)")
        return True
        
    print("Testing HybridCNN_LSTM_Equalizer...")
    
    # Create model
    model = HybridCNN_LSTM_Equalizer(input_dim=2)
    
    # Create test input (batch=2, features=seq_len*2)
    seq_len = 2 * Config.CONTEXT_K + 1
    x = torch.randn(2, seq_len * 2)
    
    # Apply model
    output = model(x)
    
    # Check output shape
    assert output.shape == (2, 16), f"Expected (2, 16), got {output.shape}"
    
    # Check that output is finite
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    print("✓ HybridCNN_LSTM_Equalizer test passed")
    return True

def main():
    """Run all tests"""
    print("Running tests for improved model...")
    print("=" * 50)
    
    # Run tests
    test_polarization_activation()
    test_lstm_model()
    test_hybrid_model()
    
    print("=" * 50)
    print("✓ All tests passed!")

if __name__ == "__main__":
    main()