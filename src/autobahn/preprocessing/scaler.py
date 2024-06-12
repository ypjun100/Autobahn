import gc
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler

def normalize(dataframe: pd.DataFrame, columns: list[str], pipeline = None):
    for column in columns:
        normalizer = Normalizer()
        dataframe[column] = normalizer.fit_transform(dataframe[column].to_numpy().reshape(1, -1)).reshape(-1, 1)

        if pipeline != None:
            pipeline.set_scaling(column, 'Normalize', normalizer)
    
    

def standardize(dataframe: pd.DataFrame, columns: list[str], pipeline = None):
    for column in columns:
        standardizer = StandardScaler()
        dataframe[column] = standardizer.fit_transform(dataframe[column].to_numpy().reshape(-1, 1))

        if pipeline != None:
            pipeline.set_scaling(column, 'Standardize', standardizer)