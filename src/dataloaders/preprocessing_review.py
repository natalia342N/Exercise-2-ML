import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_amazon_review_data(
    train_path='/kaggle/input/184-702-tu-ml-2025-s-reviews/amazon_review_ID.shuf.lrn.csv',
    test_path='/kaggle/input/184-702-tu-ml-2025-s-reviews/amazon_review_ID.shuf.tes.csv'
):
    data = pd.read_csv(train_path)


    X = data.drop(columns=['ID', 'Class'])
    y = data['Class']

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    test_data = pd.read_csv(test_path)
    ids_test = test_data['ID']
    X_test_raw = test_data.drop(columns=['ID'])
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, X_val_scaled, y_train.reset_index(drop=True), y_val.reset_index(drop=True), X_test_scaled, ids_test
