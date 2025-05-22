import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_amazon_review_data(path='datasets/review/amazon_review_ID.shuf.lrn.csv'):

    data = pd.read_csv(path)

    X = data.drop(columns=['ID', 'Class'])
    y = data['Class']

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)



    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    return X_train_scaled, X_val_scaled, y_train, y_val
