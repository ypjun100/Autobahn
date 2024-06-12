import pickle
import pandas as pd

class Pipeline:
    def __init__(self, dataframe: pd.DataFrame, target: str):
        self.data = {'target': '', 'columns': {}}
        for column in dataframe.columns:
            if column == target:
                self.data['target'] = target
                continue
            self.data['columns'][column] = {"type": "", "scaling": "False", "normalizer": None, "standardizer": None, "encoding": "False", "encoder": None, "category": []}
            if dataframe[column].dtypes in ['bool', 'category']:
                self.data['columns'][column]['type'] = 'category'
                self.data['columns'][column]['category'] = self.get_category(dataframe[column])
            else:
                self.data['columns'][column]['type'] = 'numeric'
    
    def get_category(self, series: pd.Series):
        return list(map(str, series.unique()))
    
    def set_scaling(self, column: str, scaling_method: str, scaler: object):
        if scaling_method == 'Normalize':
            self.data['columns'][column]['scaling'] = scaling_method
            self.data['columns'][column]['normalizer'] = scaler
        elif scaling_method == 'Standardize':
            self.data['columns'][column]['scaling'] = scaling_method
            self.data['columns'][column]['normalizer'] = scaler

    def set_encoding(self, column: str, encoder: object):
        self.data['columns'][column]['encoding'] = "True"
        self.data['columns'][column]['encoder'] = encoder
    
    def get_pipeline(self):
        return self.data
    
    def open(self, filename):
        with open(filename + '.pkl', 'rb') as fw:
            self.data = pickle.load(fw)

    def save(self, filename):
        with open(filename + '.pkl', 'wb') as fw:
            pickle.dump(self.data, fw)