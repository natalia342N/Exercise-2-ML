# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# class PyTorchNN:
#     def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01):
#         self.model = FeedforwardNN(input_size, hidden_layers, output_size, activation)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.input_size = input_size

#     def fit(self, X, y, epochs=100, batch_size=32):
#         self.model.train()
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.long)

#         for epoch in range(epochs):
#             permutation = torch.randperm(X_tensor.size()[0])
#             for i in range(0, X_tensor.size()[0], batch_size):
#                 indices = permutation[i:i+batch_size]
#                 batch_X = X_tensor[indices]
#                 batch_y = y_tensor[indices]

#                 self.optimizer.zero_grad()
#                 outputs = self.model(batch_X)
#                 loss = self.loss_fn(outputs, batch_y)
#                 loss.backward()
#                 self.optimizer.step()

#     def predict(self, X):
#         self.model.eval()
#         with torch.no_grad():
#             X_tensor = torch.tensor(X, dtype=torch.float32)
#             outputs = self.model(X_tensor)
#             return torch.argmax(outputs, dim=1).numpy()

#     def score(self, X, y):
#         preds = self.predict(X)
#         return (preds == y).mean()

#     def count_parameters(self):
#         return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

#     def estimate_ram_MB(self, input_size=None):
#         if input_size is None:
#             input_size = self.input_size
#         return estimate_virtual_ram_usage(self.model, input_size)


# class FeedforwardNN(nn.Module):
#     def __init__(self, input_size, hidden_layers, output_size, activation='relu'):
#         super(FeedforwardNN, self).__init__()
#         self.activation = activation.lower()

#         # Dynamically build layers
#         layers = []
#         in_dim = input_size
#         for h_dim in hidden_layers:
#             layers.append(nn.Linear(in_dim, h_dim))
#             in_dim = h_dim
#         self.hidden_layers = nn.ModuleList(layers)
#         self.output_layer = nn.Linear(in_dim, output_size)

#     def forward(self, x):
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = self._apply_activation(x)
#         return self.output_layer(x)  # raw logits

#     def _apply_activation(self, x):
#         if self.activation == 'relu':
#             return F.relu(x)
#         elif self.activation == 'sigmoid':
#             return torch.sigmoid(x)
#         elif self.activation == 'tanh':
#             return torch.tanh(x)
#         else:
#             raise ValueError(f"Unsupported activation: {self.activation}")

# def count_learnable_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def estimate_virtual_ram_usage(model, input_size, dtype=torch.float32):
#     # Estimate memory usage of weights and one forward pass
#     param_mem = sum(p.numel() * torch.tensor([], dtype=dtype).element_size()
#                     for p in model.parameters() if p.requires_grad)

#     batch = torch.zeros((1, input_size), dtype=dtype)
#     with torch.no_grad():
#         output = model(batch)
#     activation_mem = batch.numel() * batch.element_size() + output.numel() * output.element_size()

#     total_bytes = param_mem + activation_mem
#     return total_bytes / (1024 ** 2)  # in MB

# # Optional: testing if this script is run directly
# if __name__ == "__main__":
#     model = FeedforwardNN(input_size=16, hidden_layers=[32, 16], output_size=2, activation='relu')
#     print(model)
#     print("Learnable parameters:", count_learnable_params(model))
#     print("Estimated VRAM usage (MB):", estimate_virtual_ram_usage(model, input_size=16))
