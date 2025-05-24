from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import numpy as np


class SklearnNN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            learning_rate_init=learning_rate,
            max_iter=100,
            batch_size=32,
            verbose=False,
            random_state=42
        )
        self.loss_curve_ = None

    def fit(self, X, y, **kwargs):
        if len(np.unique(y)) > 2:
            y = y.flatten()
        self.model.fit(X, y)
        self.loss_curve_ = self.model.loss_curve_  # ✅ save loss


    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def count_parameters(self):
        return sum(w.size for w in self.model.coefs_) + sum(b.size for b in self.model.intercepts_)

    def estimate_ram_MB(self, input_size=None):
        bytes_per_float = 8
        weight_bytes = sum(w.size for w in self.model.coefs_) * bytes_per_float
        bias_bytes = sum(b.size for b in self.model.intercepts_) * bytes_per_float
        return (weight_bytes + bias_bytes) / (1024 ** 2)
