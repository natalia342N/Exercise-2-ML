import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from scipy import stats

def preprocess_weather_data(path='datasets/Dataset3_Weather/Weather Training Data.csv'):
    df = pd.read_csv(path)
    df.drop(columns=['row ID'], inplace=True)

    X_raw = df.drop("RainTomorrow", axis=1)
    X_raw=X_raw[:20000]
    y = df["RainTomorrow"].iloc[:20000]

    X_train_raw, X_test_raw, y_train, y_test = sk.model_selection.train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    categorical_cols = ['Location', 'RainToday', 'WindGustDir', 'WindDir3pm', 'WindDir9am']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train_raw[categorical_cols])

    X_train_encoded = encoder.transform(X_train_raw[categorical_cols])
    X_test_encoded = encoder.transform(X_test_raw[categorical_cols])

    encoded_col_names = encoder.get_feature_names_out(categorical_cols)
    X_train_cat = pd.DataFrame(X_train_encoded, columns=encoded_col_names, index=X_train_raw.index)
    X_test_cat = pd.DataFrame(X_test_encoded, columns=encoded_col_names, index=X_test_raw.index)

    X_train_num = X_train_raw.drop(columns=categorical_cols)
    X_test_num = X_test_raw.drop(columns=categorical_cols)

    X_train_combined = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_combined = pd.concat([X_test_num, X_test_cat], axis=1)


    imputer = KNNImputer(n_neighbors=3, weights='uniform')
    X_train_imputed = imputer.fit_transform(X_train_combined)
    X_test_imputed = imputer.transform(X_test_combined)

    X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_train_combined.columns)
    X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X_test_combined.columns)

    columns_to_normalize = [
        "MinTemp", "MaxTemp", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
        "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"
    ]

    X_train_imputed_df[columns_to_normalize] = X_train_imputed_df[columns_to_normalize].apply(stats.zscore)
    for col in columns_to_normalize:
        mean = X_train_imputed_df[col].mean()
        std = X_train_imputed_df[col].std()
        X_test_imputed_df[col] = (X_test_imputed_df[col] - mean) / std


    return (
        X_train_imputed_df.to_numpy(),
        X_test_imputed_df.to_numpy(),
        y_train.reset_index(drop=True).to_numpy().reshape(-1),
        y_test.reset_index(drop=True).to_numpy().reshape(-1)
    )
