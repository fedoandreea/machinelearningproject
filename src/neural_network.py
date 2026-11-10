"""M2: Feed-forward neural network implemented from scratch (standard library, NumPy, SciPy only).

TODO (owner: ____):
- FeedForwardNNScratch(hidden_sizes, learning_rate, epochs, batch_size, seed)
- Forward pass (e.g. ReLU hidden layers, linear output), MSE loss.
- Backpropagation + (mini-batch) gradient descent.
- Standardise inputs (and target) before training.
- Check correctness: gradient check with finite differences, compare with
  sklearn.neural_network.MLPRegressor or PyTorch.
"""
import numpy as np


class FeedForwardNNScratch:
    def __init__(self, hidden_sizes=(32, 32), learning_rate=1e-3, epochs=500, batch_size=32, seed=42):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
