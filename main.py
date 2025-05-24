import itertools
import numpy as np
import pandas as pd

# preprocessing 
from datasets.preprocessing_review import preprocess_amazon_review_data
from datasets.preprocessing_weather import preprocess_weather_data

# NN models 
from framework.my_nn import MyNN
from framework.LLM_nn import LLM_NN
from framework.pytorch_nn import PyTorchNN
from framework.sklearn_nn import SklearnNN

# training and evaluation
from training.train import train_model
from training.evaluate import evaluate_model

# debug data
from datasets.debug_data import load_debug_data  

# pick debug, amazon review or weather dataset
X_train, X_test, y_train, y_test = preprocess_weather_data()

input_size=X_train.shape[1]
output_size = len(np.unique(y_train))

param_grid = {
    'hidden_layers': [
        (16,),        # 1 layer, 8 neurons
        (32,),       # 1 layer, 32 neurons
        (64, 32),    # 2 layers - 64 and 32 neurons
        (64, 32, 16) # 3 layers - ...
    ],
    'learning_rate': [0.001, 0.01],
    'activation': ['relu', 'tanh']
}

base_models = {
    "MyNN": MyNN,
    "LLM_NN": LLM_NN,
    "PyTorchNN": PyTorchNN,
    "SklearnNN": SklearnNN
}

results = {}
best_models = {}
log = [] 

for model_name, model_class in base_models.items():
    print(f" Grid Search for {model_name}")
    best_score = -np.inf
    best_model = None
    best_params = None

    for hidden_layers, learning_rate, activation in itertools.product(
            param_grid['hidden_layers'],
            param_grid['learning_rate'],
            param_grid['activation']):

        print(f"  Trying hidden_layers={hidden_layers}, learning_rate={learning_rate}, activation={activation}")

        try:
            model = model_class(
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=output_size,
                activation=activation,
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
        log.append({
            'model': model_name,
            'hidden_layers': hidden_layers,
            'learning_rate': learning_rate,
            'activation': activation,
            'accuracy': acc
        })


        if acc > best_score:
            best_score = acc
            best_model = model
            best_params = {
                'hidden_layers': hidden_layers,
                'learning_rate': learning_rate,
                'activation': activation
            }

            results[model_name] = result

    print(f" Best for {model_name}: {best_params}, Accuracy={best_score:.4f}")
    best_models[model_name] = best_model

print("\n=== Final Results ===")
for name, result in results.items():
    print(f"{name}: Accuracy={result['accuracy']:.4f}, Params={result['params']}, RAM={result['ram_MB']:.2f} MB")
df_results = pd.DataFrame(log)

# adjust .csv output name 
df_results.to_csv("grid_search_results_weather_all.csv", index=False)
print("\nSaved full grid search results to grid_search_results... .csv")