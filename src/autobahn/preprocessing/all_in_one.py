import pandas as pd
# from autobahn.preprocessing import column_cleaner
# from autobahn.preprocessing import type_converter

import column_cleaner
import type_converter
import missing_value

# Run all steps in preprocessing
def run(dataframe: pd.DataFrame, target: str):
    column_cleaner.run(dataframe)
    type_converter.run(dataframe)
    missing_value.run(dataframe, target)

# df = pd.read_csv("/home/jovyan/work/Autobahn/dataset/test.csv", na_values=missing_value.MISSING_VALUE_SYMBOL)
# df = pd.read_csv("/home/jovyan/work/Autobahn/dataset/data_csv_test.csv", na_values=missing_value.MISSING_VALUE_SYMBOL)
# run(df, 'ASD_traits')
# print(df)
# print(df.isna().sum())