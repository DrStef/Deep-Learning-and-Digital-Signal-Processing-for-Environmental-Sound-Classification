import pytest
import numpy as np
from src.models.stage2_cnn import build_cnn_model, train_cnn_model

def test_build_cnn_model():
    model = build_cnn_model(input_shape=(128, 128, 3), num_classes=10)
    assert len(model.layers) >= 10  # Conv + Pooling + Dense
    assert model.output_shape == (None, 10)

def test_train_cnn_model():
    model = build_cnn_model((128, 128, 3), 10)
    train_X = np.random.randn(10, 128, 128, 3)
    train_y = np.eye(10)[np.random.randint(0, 10, 10)]
    test_X = np.random.randn(2, 128, 128, 3)
    test_y = np.eye(10)[np.random.randint(0, 10, 2)]
    history = train_cnn_model(model, train_X, train_y, test_X, test_y, epochs=1)
    assert 'accuracy' in history.history