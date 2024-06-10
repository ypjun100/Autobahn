import gc
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler

def normalize(dataframe: pd.DataFrame, columns: list[str]):
    normalizer = Normalizer()
    for column in columns:
        dataframe[column] = normalizer.fit_transform(dataframe[column].to_numpy().reshape(1, -1)).reshape(-1, 1)
    del normalizer
    gc.collect()

def standardize(dataframe: pd.DataFrame, columns: list[str]):
    standardizer = StandardScaler()
    for column in columns:
        dataframe[column] = standardizer.fit_transform(dataframe[column].to_numpy().reshape(-1, 1))
    del standardizer
    gc.collect()