import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Find category columns in dataframe
def detect(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if dataframe[column].dtypes == 'category']


# Encode category column
def encode_category(dataframe: pd.DataFrame, column: str, pipeline = None):
    dataframe[column] = pd.Series(list(map(str.upper, map(str, dataframe[column])))).astype('category')
    le = LabelEncoder()
    dataframe[column] = le.fit_transform(dataframe[column])

    if pipeline != None:
        pipeline.set_encoding(column, le)


# Main runner
def run(dataframe: pd.DataFrame, target: str, pipeline = None):
    for column in detect(dataframe):
        if column != target:
            encode_category(dataframe, column, pipeline)