import itertools

import numpy as np

from datasets.preprocessing_review import preprocess_amazon_review_data
from datasets.preprocessing_weather import preprocess_weather_data
from framework.my_nn import MyNN
from framework.LLM_nn import LLM_NN
# from framework.pytorch_nn import PyTorchNN
from framework.sklearn_nn import SklearnNN
from training.train import train_model
from training.evaluate import evaluate_model
from datasets.debug_data import load_debug_data  # Use debug for now

# Load test data
X_train, X_test, y_train, y_test = preprocess_amazon_review_data()
input_size=X_train.shape[1]
output_size = len(np.unique(y_train))

# Define hyperparameter grid
param_grid = {
    'hidden_layers': [(8,), (32,), (64, 32)],
    'learning_rate': [0.001, 0.01]
}

base_models = {
    "MyNN": MyNN,
    #"LLM_NN": LLM_NN,
    #"PyTorchNN": PyTorchNN,
    "SklearnNN": SklearnNN
}

results = {}
best_models = {}

for model_name, model_class in base_models.items():
    print(f" Grid Search for {model_name}")
    best_score = -np.inf
    best_model = None
    best_params = None

    for hidden_layers, learning_rate in itertools.product(param_grid['hidden_layers'], param_grid['learning_rate']):
        print(f"  Trying hidden_layers={hidden_layers}, learning_rate={learning_rate}")
        try:
            model = model_class(
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=output_size,
                activation='relu',
                learning_rate=learning_rate
            )
        except TypeError:
            model = model_class(
                input_size=input_size,
                hidden_layers=hidden_layers,
                learning_rate=learning_rate
            )

        train_model(model, X_train, y_train, epochs=100, batch_size=32, verbose=False)
        result = evaluate_model(model, X_test, y_test, input_size=input_size)
        acc = result['accuracy']
        print(f"    → Accuracy: {acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_params = {'hidden_layers': hidden_layers, 'learning_rate': learning_rate}
            results[model_name] = result

    print(f" Best for {model_name}: {best_params}, Accuracy={best_score:.4f}")
    best_models[model_name] = best_model

# Print final results
print("\n=== Final Results ===")
for name, result in results.items():
    print(f"{name}: Accuracy={result['accuracy']:.4f}, Params={result['params']}, RAM={result['ram_MB']:.2f} MB")
