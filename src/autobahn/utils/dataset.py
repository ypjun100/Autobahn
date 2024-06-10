import pandas as pd

# Return statistics of dataset
# Parameter, type, unique, number of na, mean
def get_dataset_statistics(dataframe):
    statistics = pd.DataFrame({'Feature Name': dataframe.columns.to_list(),
                        'Type': [dataframe[col].dtypes for col in dataframe.columns],
                        'Unique': [str(len(dataframe[col].unique())) if dataframe[col].dtypes in ['bool', 'category'] else '-' for col in dataframe.columns],
                        'Missing values': dataframe.isnull().sum().to_list(),
                        'Mean': [str(round(dataframe[col].mean(), 5)) if dataframe[col].dtypes not in ['bool', 'category'] else '-' for col in dataframe.columns]})
    return statistics


# Verify dataset
def verify(dataframe: pd.DataFrame, target="") -> tuple[bool, str]:
    # Error : dataset is empty (number of column is less than two)
    if len(dataframe.columns) < 2:
        return (False, 'Cannot load dataset with less than two columns.')

    # Error : type of target column is not category
    if target != "":
        if int(len(dataframe) * 0.2) <= len(dataframe[target].unique()):
            return (False, "There are so many categories in target column.")
    return (True, '')