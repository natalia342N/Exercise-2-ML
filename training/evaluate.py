def evaluate_model(model, X_test, y_test, input_size=None):
    results = {}
    try:
        accuracy = model.score(X_test, y_test)
        results['accuracy'] = accuracy
    except Exception as e:
        results['accuracy'] = None
        print(f"Error scoring model {model.__class__.__name__}: {e}")

    try:
        params = model.count_parameters()
        results['params'] = params
    except Exception as e:
        results['params'] = None
        print(f"Error counting parameters for {model.__class__.__name__}: {e}")

    try:
        if input_size is None:
            input_size = X_test.shape[1]
        ram = model.estimate_ram_MB(input_size=input_size)
        results['ram_MB'] = ram
    except Exception as e:
        results['ram_MB'] = None
        print(f"Error estimating RAM for {model.__class__.__name__}: {e}")

    return results
