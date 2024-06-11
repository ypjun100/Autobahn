import pandas as pd

# TODO: There is a problem in detecting outliers.
# If there is a cateory with small rate in entire cateogies, this code detects that category as an outlier.

def get_outlier_indexes(series: pd.Series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return series[(series < lower_bound) | (series > upper_bound)].index


def run_for_dataframe(dataframe: pd.DataFrame):
    for column in dataframe.columns:
        if 'int' in str(dataframe[column].dtypes).lower() or\
            'float' in str(dataframe[column].dtypes).lower():
            outlier_indexes = get_outlier_indexes(dataframe[column])
            dataframe.drop(outlier_indexes, inplace=True)