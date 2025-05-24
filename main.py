import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets.preprocessing_weather import preprocess_weather_data
from datasets.preprocessing_review import preprocess_amazon_review_data
from framework.my_nn import MyNN
from framework.LLM_nn import LLM_NN
from framework.pytorch_nn import PyTorchNN
from framework.sklearn_nn import SklearnNN
from training.evaluate import evaluate_model
from training.train import train_model

X_train, X_test, y_train, y_test = preprocess_weather_data()
input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

param_grid = {
    'hidden_layers': [(16,), (32,), (64, 32), (64, 32, 16)],
    'learning_rate': [0.001, 0.01],
    'activation': ['relu', 'tanh']

}

base_models = {
    # "MyNN": MyNN,
    # "LLM_NN": LLM_NN,
    # "PyTorchNN": PyTorchNN,
    "SklearnNN": SklearnNN
}

results = {}
log = []
best_losses_mynn = []
best_losses_llm = []
best_losses_pytorch = []
best_losses_sklearn = []

for model_name, model_class in base_models.items():
    print(f"\n Grid Search for {model_name}")
    best_score = -np.inf
    best_model = None

    for hidden_layers, learning_rate, activation in itertools.product(
        param_grid['hidden_layers'], param_grid['learning_rate'], param_grid['activation']
    ):
        print(f"  Trying hidden={hidden_layers}, lr={learning_rate}, act={activation}")
        model = model_class(input_size, hidden_layers, output_size, learning_rate=learning_rate, activation=activation)

        if model_name == "MyNN":
            losses = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=False)
        elif model_name == "LLM_NN":
            losses = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=False)
        elif model_name == "PyTorchNN":
            losses = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=False)
        elif model_name == "SklearnNN":
            model.fit(X_train, y_train)
            losses = model.loss_curve_

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
            results[model_name] = result

            if model_name == "MyNN":
                best_losses_mynn = losses
            elif model_name == "LLM_NN":
                best_losses_llm = losses
            elif model_name == "PyTorchNN":
                best_losses_pytorch = losses
            elif model_name == "SklearnNN":
                best_losses_sklearn = losses

pd.DataFrame({'loss': best_losses_mynn}).to_csv("losses_mynn.csv", index=False)
pd.DataFrame({'loss': best_losses_llm}).to_csv("losses_llm.csv", index=False)
pd.DataFrame({'loss': best_losses_pytorch}).to_csv("losses_pytorch.csv", index=False)
pd.DataFrame({'loss': best_losses_sklearn}).to_csv("losses_sklearn.csv", index=False)

df_results = pd.DataFrame(log)
df_results.to_csv("grid_search_results_weather.csv", index=False)

print("\n=== Final Results Summary ===")
for name, result in results.items():
    print(f"{name}: Accuracy={result['accuracy']:.4f}, Params={result['params']}, RAM={result['ram_MB']:.2f} MB")



plt.figure(figsize=(10, 6))
if best_losses_mynn: plt.plot(best_losses_mynn, label="MyNN")
if best_losses_llm: plt.plot(best_losses_llm, label="LLM_NN")
if best_losses_pytorch: plt.plot(best_losses_pytorch, label="PyTorchNN")
if best_losses_sklearn: plt.plot(best_losses_sklearn, label="SklearnNN")

plt.title("Convergence Curves (Best Configurations)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("convergence_all_models.png")
plt.show()
