import pickle
import pandas as pd

class Pipeline:
    def __init__(self, dataframe: pd.DataFrame, target: str):
        self.data = {}
        for column in dataframe.columns:
            if column == target:
                continue
            self.data[column] = {"type": "", "scaling": "False", "normalizer": None, "standardizer": None, "encoding": "False", "encoder": None, "category": []}
            if dataframe[column].dtypes in ['bool', 'category']:
                self.data[column]['type'] = 'category'
            else:
                self.data[column]['type'] = 'numeric'
    
    def set_scaling(self, column: str, scaling_method: str, scaler: object):
        if scaling_method == 'Normalize':
            self.data[column]['scaling'] = scaling_method
            self.data[column]['normalizer'] = scaler
        elif scaling_method == 'Standardize':
            self.data[column]['scaling'] = scaling_method
            self.data[column]['standardizer'] = scaler

    def set_encoding(self, column: str, encoder: object, category: list):
        self.data[column]['encoding'] = "True"
        self.data[column]['encoder'] = encoder
        self.data[column]['category'] = category
    
    def get_pipeline(self):
        return self.data
    
    @staticmethod
    def open(filename):
        with open(filename + '.pkl', 'rb') as fw:
            return pickle.load(fw)

    def save(self, filename):
        with open(filename + '.pkl', 'wb') as fw:
            pickle.dump(self.data, fw)