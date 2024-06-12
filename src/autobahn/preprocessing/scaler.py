import numpy as np
import pandas as pd


class Normalizer:
    def __init__(self):
        self.x_min = 0
        self.x_max = 0
    
    def fit_transform(self, arr: list):
        self.x_min = min(arr)
        self.x_max = max(arr)
        return (arr - self.x_min) / (self.x_max - self.x_min)
    
    def transform(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)


class Standardizer:
    def __init__(self):
        self.mu = 0
        self.sigma = 0
    
    def fit_transform(self, arr: np.array):
        _arr = arr[~np.isnan(arr)]
        self.mu = np.mean(_arr)
        self.sigma = np.std(_arr)
        return (arr - self.mu) / self.sigma
    
    def transform(self, x):
        return (x - self.mu) / self.sigma


def normalize(dataframe: pd.DataFrame, columns: list[str], pipeline = None):
    for column in columns:
        normalizer = Normalizer()
        dataframe[column] = normalizer.fit_transform(dataframe[column].to_numpy()).reshape(-1, 1)

        if pipeline != None:
            pipeline.set_scaling(column, 'Normalize', normalizer)
    
    

def standardize(dataframe: pd.DataFrame, columns: list[str], pipeline = None):
    for column in columns:
        standardizer = Standardizer()
        dataframe[column] = standardizer.fit_transform(dataframe[column].to_numpy())

        if pipeline != None:
            pipeline.set_scaling(column, 'Standardize', standardizer)