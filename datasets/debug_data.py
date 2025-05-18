from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_debug_data(n_samples=10, n_features=4, n_classes=2):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=1)
