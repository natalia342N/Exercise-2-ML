from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class SklearnNN:
    def __init__(self, input_size=None, hidden_layers=(100,), activation='relu', output_size=None, learning_rate=0.001):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=500,
            random_state=0,
            verbose=False 
        )

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)

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
