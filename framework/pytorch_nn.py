import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        if isinstance(y, np.ndarray):
            if y.ndim == 2 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            elif y.ndim == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            y = torch.tensor(y, dtype=torch.long)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class PyTorchNN:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01):
        self.model = SampleClassNN(input_size, hidden_layers, output_size, activation)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.input_size = input_size

    def fit(self, X, y, epochs=100, batch_size=32, verbose=False):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)  # One-hot → class index
        y_tensor = torch.tensor(y, dtype=torch.long)

        losses = []  # 📌 Store epoch loss

        for epoch in range(epochs):
            permutation = torch.randperm(X_tensor.size()[0])
            epoch_loss = 0
            num_batches = 0

            for i in range(0, X_tensor.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_X = X_tensor[indices]
                batch_y = y_tensor[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return losses  # 📤 Return tracked losses


                
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            return torch.argmax(outputs, dim=1).numpy()

    def score(self, X, y):
        preds = self.predict(X)
        return (preds == y).mean()

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def estimate_ram_MB(self, input_size=None):
        if input_size is None:
            input_size = self.input_size
        return estimate_virtual_ram_usage(self.model, input_size)


class SampleClassNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation='relu'):
        super(SampleClassNN, self).__init__()
        self.activation = activation.lower()

        layers = []
        in_dim = input_size
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_dim, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self._apply_activation(x)
        return self.output_layer(x)

    def _apply_activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

def count_learnable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_virtual_ram_usage(model, input_size, dtype=torch.float32):
    param_mem = sum(p.numel() * torch.tensor([], dtype=dtype).element_size()
                    for p in model.parameters() if p.requires_grad)

    batch = torch.zeros((1, input_size), dtype=dtype)
    with torch.no_grad():
        output = model(batch)
    activation_mem = batch.numel() * batch.element_size() + output.numel() * output.element_size()

    total_bytes = param_mem + activation_mem
    return total_bytes / (1024 ** 2) 
