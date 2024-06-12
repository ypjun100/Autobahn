import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Encoder:
    def __init__(self):
        self.mapper = {}
        self.categories = []
    
    def fit_transform(self, series: pd.Series):
        self.categories = sorted(list(map(str, series.dropna().unique())))
        
        index = 0
        for category in self.categories:
            is_union = False
            for key in self.mapper:
                if key.upper() == category.upper():
                    self.mapper[category] = self.mapper[key]
                    is_union = True
                    break
            
            if not is_union:
                self.mapper[category] = str(index)
                index += 1
        
        result = []
        for i in range(len(series)):
            result.append(self.mapper[str(series[i])])

        return pd.Series(result).astype('category')
    
    def transform(self, x: str):
        return self.mapper[str(x)]



# Find category columns in dataframe
def detect(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if dataframe[column].dtypes == 'category']


# Encode category column
def encode_category(dataframe: pd.DataFrame, column: str, pipeline = None):
    encoder = Encoder()
    dataframe[column] = encoder.fit_transform(dataframe[column])

    if pipeline != None:
        pipeline.set_encoding(column, encoder)


# Main runner
def run(dataframe: pd.DataFrame, target: str, pipeline = None):
    for column in detect(dataframe):
        if column != target:
            encode_category(dataframe, column, pipeline)