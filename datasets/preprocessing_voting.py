import pandas as pd
import numpy as np
from dbfread import DBF
from sklearn.model_selection import train_test_split

def load_dataset(config):
    dfs = []
    for path in config['dbf_paths']:
        table = DBF(path)
        dfs.append(pd.DataFrame(iter(table)))

    data = pd.concat(dfs, ignore_index=True)
    data.rename(columns={
        'BAND_1': 'Blue', 'BAND_2': 'Green',
        'BAND_3': 'Red', 'BAND_4': 'NIR'
    }, inplace=True)

    data['NDVI'] = (data['NIR'] - data['Red']) / (data['NIR'] + data['Red'])

    X = data[['Blue', 'Green', 'Red', 'NIR', 'NDVI']].astype(float).values
    Y = data[config['response_var']].astype(float).values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Normalize
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test
