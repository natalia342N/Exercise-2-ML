import numpy as np

class LLM_NN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01):
        self.layers = [input_size] + list(hidden_layers) + [output_size]

        self.activation = activation
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2. / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")

    def _activation_deriv(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1.0 - np.tanh(x)**2
        else:
            raise ValueError("Unsupported activation function")

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, probs, y_true):
        n = y_true.shape[0]
        correct_probs = probs[range(n), y_true]
        loss = -np.sum(np.log(correct_probs + 1e-8)) / n
        return loss

    def _cross_entropy_grad(self, probs, y_true):
        n = y_true.shape[0]
        grad = probs.copy()
        grad[range(n), y_true] -= 1
        grad /= n
        return grad

    def forward(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            zs.append(z)
            a = self._activation(z)
            activations.append(a)

        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        zs.append(z)
        a = self._softmax(z)
        activations.append(a)
        return zs, activations

    def backward(self, zs, activations, y_true):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Output layer
        delta = self._cross_entropy_grad(activations[-1], y_true)
        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Hidden layers
        for l in reversed(range(len(self.weights) - 1)):
            delta = (delta @ self.weights[l + 1].T) * self._activation_deriv(zs[l])
            grads_w[l] = activations[l].T @ delta
            grads_b[l] = np.sum(delta, axis=0, keepdims=True)

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b, max_norm=5.0):
        for i in range(len(self.weights)):
            grad_norm = np.linalg.norm(grads_w[i])
            if grad_norm > max_norm:
                grads_w[i] = grads_w[i] * (max_norm / grad_norm)
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]


    def fit(self, X, y, epochs=100, batch_size=32, verbose=False):
        losses = []  # 📌 Track per-epoch loss

        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            num_batches = 0

            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                zs, activations = self.forward(X_batch)
                loss = self._cross_entropy_loss(activations[-1], y_batch)
                grads_w, grads_b = self.backward(zs, activations, y_batch)
                self.update_params(grads_w, grads_b)

                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return losses  # 📤 Return list of per-epoch losses
            

    def predict(self, X):
        _, activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def count_parameters(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def estimate_ram_MB(self, input_size):
        bytes_per_float = 8
        activations_mem = sum(w.shape[0] * bytes_per_float for w in self.weights)
        weights_mem = sum(w.size * bytes_per_float for w in self.weights)
        biases_mem = sum(b.size * bytes_per_float for b in self.biases)
        input_mem = input_size * bytes_per_float
        total = activations_mem + weights_mem + biases_mem + input_mem
        return total / (1024**2)
