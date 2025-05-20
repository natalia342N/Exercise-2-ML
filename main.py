from datasets.preprocessing_weather import preprocess_weather_data
from framework.my_nn import MyNN
from framework.LLM_nn import LLM_NN
# from framework.pytorch_nn import PyTorchNN
from framework.sklearn_nn import SklearnNN
from training.train import train_model
from training.evaluate import evaluate_model
from datasets.debug_data import load_debug_data  # Use debug for now

# Load small test data
X_train, X_test, y_train, y_test = preprocess_weather_data()

# Model definitions (hardcoded)
X_train_shape=X_train.shape[1]
models = {
    "MyNN": MyNN(input_size=X_train_shape, hidden_layers=[8], output_size=1,  learning_rate=0.01),
    #"LLM_NN": LLM_NN(input_size=X_train_shape, hidden_layers=[8], output_size=1, activation='relu', learning_rate=0.01),
    # "PyTorchNN": PyTorchNN(input_size=4, hidden_layers=[8], output_size=2, activation='relu', learning_rate=0.01),
    "SklearnNN": SklearnNN(input_size=X_train_shape, hidden_layers=(8,), output_size=1, activation='relu', learning_rate=0.01)
}



# Train and evaluate all models
results = {}
for name, model in models.items():
    # print(f"\n=== Training {name} ===")
    # train_model(model, X_train, y_train, epochs=100, batch_size=32, verbose=True)
    print(f"Training {name} with verbose=True")
    train_model(model, X_train, y_train, epochs=100, batch_size=32, verbose=True)


    results[name] = evaluate_model(model, X_test, y_test, input_size=X_test.shape[1])
    

# Print results
print("\n=== Final Results ===")
for name, result in results.items():
    print(f"{name}: Accuracy={result['accuracy']:.4f}, Params={result['params']}, RAM={result['ram_MB']:.2f} MB")


