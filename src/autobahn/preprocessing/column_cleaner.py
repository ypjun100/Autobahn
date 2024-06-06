import pandas as pd

# Remove columns which don't have any information
def run(dataframe: pd.DataFrame):
    for column in dataframe.columns:
        # If column have only na values, delete column
        if sum(dataframe[column].isna()) == len(dataframe):
            dataframe.drop(column, axis=1, inplace=True)
            continue

        # Delete columns which have one unique value
        if len(dataframe[column].dropna(axis=0).unique()) == 1:
            dataframe.drop(column, axis=1, inplace=True)

        # If ratio of missing values in column is over 90%, delete column
        if int(len(dataframe) * 0.9) <= sum(dataframe[column].isna()):
            dataframe.drop(column, axis=1, inplace=True)
    
    # Delete columns which don't have name
    for column in dataframe.columns[dataframe.columns.str.contains('unnamed',case = False)]:
        dataframe.drop(column, axis=1, inplace=True)