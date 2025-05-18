# From scratch version of a neural network

import numpy as np

class MyNN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01):
        self.net = Network(input_size, hidden_layers + [output_size])
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        
        # Convert y to one-hot if necessary (for multi-class)
        if y.ndim == 1:
            y_encoded = self._one_hot(y)
        else:
            y_encoded = y
        self.net.train_network(
            learning_rate=self.learning_rate,
            X_train=X,
            Y_train=y_encoded,
            epoch_n=epochs,
            batch_size=batch_size,
            verbose=verbose  # ✅ Add this
        )

    def predict(self, X):
        output = self.net.forwardPropagation(X)
        if output.shape[1] == 1:
            return (output > 0.5).astype(int).flatten()
        else:
            return np.argmax(output, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def count_parameters(self):
        return sum(w.size + b.size for w, b in self.net.get_all_weights())

    def estimate_ram_MB(self, input_size=None):
        bytes_per_float = 8
        total = self.count_parameters() * bytes_per_float
        return total / (1024 ** 2)

    def _one_hot(self, y):
        classes = np.unique(y)
        y_one_hot = np.zeros((y.shape[0], len(classes)))
        for idx, cls in enumerate(classes):
            y_one_hot[y == cls, idx] = 1
        return y_one_hot
    
    
class Layer:
    def __init__(self, input_n, output_n, activationFunction):
        self.weights = np.random.randn(input_n, output_n) * 0.01
        self.biases = np.zeros((1, output_n))
        self.activation = activationFunction

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def forwardPropagation(self, input_data):
        self.inputdata = input_data
        self.z = np.dot(self.inputdata, self.weights) + self.biases

        if self.activation == "relu":
            self.output = self.relu(self.z)
        elif self.activation == "sigm":
            self.output = self.sigmoid(self.z)
        else:
            raise ValueError("Unsupported activation function")

        return self.output

    def backwardPropagation(self, delta_val, learningrate):
        if self.activation == "relu":
            delta = delta_val * self.relu_derivative(self.output)
        elif self.activation == "sigm":
            delta = delta_val * self.sigmoid_derivative(self.output)

        weight_grad = np.dot(self.inputdata.T, delta)
        bias_grad = np.sum(delta, axis=0, keepdims=True)

        self.weights -= learningrate * weight_grad
        self.biases -= learningrate * bias_grad

        return np.dot(delta, self.weights.T)

    def get_weights(self):
        return (self.weights.copy(), self.biases.copy())

    def set_weights(self, weights):
        self.weights, self.biases = weights


class Network:
    def __init__(self, input_len, output_len):
        self.layers = []
        self.best_val_loss = float("inf")
        self.best_weights = None
        self.build_network(input_len, output_len)

    def build_network(self, input_len, output_len):
        self.layers.append(Layer(input_len, output_len[0], "relu"))
        for i in range(1, len(output_len)):
            activation = "relu" if i < len(output_len) - 1 else "sigm"
            self.layers.append(Layer(output_len[i - 1], output_len[i], activation))

    def forwardPropagation(self, x):
        for layer in self.layers:
            x = layer.forwardPropagation(x)
        return x

    def compute_loss(self, Y, output):
        epsilon = 1e-15
        output = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.mean(Y * np.log(output) + (1 - Y) * np.log(1 - output))
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_weights = self.get_all_weights()
        return loss

    def backwardPropagation(self, error, learning_rate):
        delta = error
        for layer in reversed(self.layers):
            delta = layer.backwardPropagation(delta, learningrate=learning_rate)

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
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            epoch_loss = 0
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                x_batch = X_train[start:end]
                y_batch = Y_train[start:end]

                output = self.forwardPropagation(x_batch)
                loss = self.compute_loss(y_batch, output)
                error = -(y_batch / output - (1 - y_batch) / (1 - output))
                self.backwardPropagation(error, learning_rate)
                epoch_loss += loss * len(x_batch)

            avg_loss = epoch_loss / len(X_train)
            losses.append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == epoch_n - 1):
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        self.set_all_weights(self.best_weights)
        return losses
    
    
