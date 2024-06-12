import pandas as pd

class Pipeline:
    def __init__(self, dataframe: pd.DataFrame):
        self.data = {}
        for column in dataframe.columns:
            self.data[column] = {"type": "", "scaling": "False", "normalizer": None, "standardizer": None, "encoding": "False", "encoder": None, "category": []}
            if dataframe[column].dtypes in ['bool', 'category']:
                self.data[column]['type'] = 'category'
                self.data[column]['category'] = self.get_category(dataframe[column])
            else:
                self.data[column]['type'] = 'numeric'
    
    def get_category(self, series: pd.Series):
        return list(map(str, series.unique()))
    
    def set_scaling(self, column: str, scaling_method: str, scaler: object):
        if scaling_method == 'Normalize':
            self.data[column]['scaling'] = scaling_method
            self.data[column]['normalizer'] = scaler
        elif scaling_method == 'Standardize':
            self.data[column]['scaling'] = scaling_method
            self.data[column]['normalizer'] = scaler

    def set_encoding(self, column: str, encoder: object):
        self.data[column]['encoding'] = "True"
        self.data[column]['encoder'] = encoder
    
    def get_pipeline(self):
        return self.data