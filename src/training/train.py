# training/train.py

def train_model(model, X_train, y_train, **kwargs):
    try:
        model.fit(X_train, y_train, **kwargs)
    except Exception as e:
        print(f"Training failed for model {model.__class__.__name__}: {e}")
        raise
    return model

