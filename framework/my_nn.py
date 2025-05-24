import numpy as np

class Activation:
    """
    A class to handle activation functions and their derivatives.
    """

    def __init__(self, name):
        self.name = name

    def forward(self, x):
        pass

    def backward(self, x):
        pass


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)

class Softmax(Activation):
    def __init__(self):
        super().__init__("softmax")

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x):
        return 1.0


class Tanh(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1.0 - np.tanh(x)**2

CLASS_MAP = {
    'relu': ReLU(),
    'sigmoid': Sigmoid(),
    'softmax': Softmax(),
    'tanh': Tanh()
}

class Layer:
    def __init__(self, input_n, output_n, activation_function: Activation):
        if activation_function == "relu":
            self.weights = np.random.randn(input_n, output_n) * np.sqrt(2.0 / input_n)
        else:
            self.weights = np.random.randn(input_n, output_n) * np.sqrt(1.0 / input_n)

        self.biases = np.zeros((1, output_n))
        self.activation = activation_function

    def forwardPropagation(self, input_data):
        self.inputdata = input_data
        self.z = np.dot(self.inputdata, self.weights) + self.biases
        self.output = self.activation.forward(self.z)
        return self.output

    def backwardPropagation(self, delta_val, learning_rate):
        delta = delta_val * self.activation.backward(self.z)
        delta = np.clip(delta, -5, 5)

        weight_grad = np.dot(self.inputdata.T, delta)
        bias_grad = np.sum(delta, axis=0, keepdims=True)

        self.weights -= learning_rate * weight_grad
        self.biases -= learning_rate * bias_grad

        return np.dot(delta, self.weights.T)

    def get_weights(self):
        return self.weights.copy(), self.biases.copy()

    def set_weights(self, weights):
        self.weights, self.biases = weights


class Network:
    def __init__(self, input_len, output_len, activation: Activation = ReLU()):
        self.layers = []
        self.best_val_loss = float("inf")
        self.best_weights = None
        self.activation = activation
        self.build_network(input_len, output_len)

    def build_network(self, input_len, output_len):
        # Add the hidden layers
        for i in range(len(output_len) - 1):
            self.layers.append(
                Layer(
                    input_len if i == 0 else output_len[i - 1],
                    output_len[i],
                    self.activation  
                )
            )

        # Add the output layer
        output_size = output_len[-1]
        output_activation = Sigmoid() if output_size == 1 else Softmax()
        self.layers.append(
            Layer(
                output_len[-2] if len(output_len) > 1 else input_len,
                output_size,
                output_activation
            )
        )

    def forwardPropagation(self, x):
        for layer in self.layers:
            x = layer.forwardPropagation(x)
        return x

    def compute_loss(self, Y, output):
        """
        Computes the cross-entropy loss value for the given predictions and true labels, applies clipping
        to avoid logarithmic errors, and updates the best validation loss and weights
        if the current loss is better.

        :param Y: Ground-truth labels for the dataset. Shape and contents depend
            on whether it's binary classification or multi-class classification.
        :param output: Model predictions after applying the forward pass. Values are
            clipped to avoid logarithmic instabilities.
        :return: The computed loss value as a float.
        :rtype: float
        """
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)

        if output.shape[1] == 1:
            # Binary Classification Case
            # Assumes:
            # - Y.shape == (N, 1) with values 0 or 1.
            # - output.shape == (N, 1) with sigmoid probabilities.
            loss = -np.mean(Y * np.log(output) + (1 - Y) * np.log(1 - output))
        else:
            # Multi-class Classification Case
            # Assumes:
            # - Y is one-hot encoded: shape (N, C)
            # - output contains softmax probabilities: shape (N, C)
            loss = -np.sum(Y * np.log(output)) / Y.shape[0]

        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_weights = self.get_all_weights()

        return loss

    def backwardPropagation(self, error, learning_rate):
        delta = error
        for layer in reversed(self.layers):
            delta = layer.backwardPropagation(delta, learning_rate=learning_rate)

    def get_all_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_all_weights(self, best_weights):
        if not isinstance(best_weights, list):
            raise TypeError("best_weights must be a list")
        if len(best_weights) != len(self.layers):
            raise ValueError("Mismatch in number of weight sets")
        for layer, weights in zip(self.layers, best_weights):
            layer.set_weights(weights)

    def train_network(self, learning_rate, X_train, Y_train, epoch_n, batch_size, verbose=True):
        losses = []
        for epoch in range(epoch_n):
            current_lr = learning_rate * (0.99 ** (epoch // 10))

            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            epoch_loss = 0
            for start in range(0, len(X_shuffled), batch_size):
                end = min(start + batch_size, len(X_shuffled))
                x_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]

                output = self.forwardPropagation(x_batch)
                loss = self.compute_loss(y_batch, output)

                error = output - y_batch
                self.backwardPropagation(error, current_lr)

                epoch_loss += loss * len(x_batch)

            avg_loss = epoch_loss / len(X_shuffled)
            losses.append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == epoch_n - 1):
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        self.set_all_weights(self.best_weights)
        return losses


class MyNN:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, activation: str = "relu"):
        self.net = Network(input_size, hidden_layers + (output_size,), activation=CLASS_MAP[activation])
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.X_mean = None
        self.X_std = None
        self.activation = activation

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_normalized = (X - self.X_mean) / (self.X_std + 1e-8)

        if self.output_size == 1:
            # Binary Encoding
            # Reshape to (N, 1) if needed
            y_encoded = y.reshape(-1, 1) if y.ndim == 1 else y
        else:
            # Multiclass Encoding
            # convert class labels into one-hot encoded vectors
            y_encoded = self._one_hot(y)

        losses = self.net.train_network(
            learning_rate=self.learning_rate,
            X_train=X_normalized,
            Y_train=y_encoded,
            epoch_n=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        return losses
        

    def predict(self, X):
        X_normalized = (X - self.X_mean) / (self.X_std + 1e-8)
        output = self.net.forwardPropagation(X_normalized)
        if self.output_size == 1:
            return (output > 0.5).astype(int).flatten()
        else:
            return np.argmax(output, axis=1)

    def score(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions == y)

    def count_parameters(self):
        return sum(w.size + b.size for w, b in self.net.get_all_weights())

    def estimate_ram_MB(self, input_size=None):
        bytes_per_float = 8
        total = self.count_parameters() * bytes_per_float
        return total / (1024 ** 2)

    def _one_hot(self, y):
        y = y.flatten()
        classes = np.unique(y)
        y_one_hot = np.zeros((y.shape[0], len(classes)))
        for idx, cls in enumerate(classes):
            y_one_hot[y == cls, idx] = 1
        return y_one_hot