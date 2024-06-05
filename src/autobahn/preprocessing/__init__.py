import pandas as pd
from autobahn.preprocessing import column_cleaner
from autobahn.preprocessing import type_converter
# import column_cleaner
# import type_converter

# Run all steps in preprocessing
def all_in_one(dataframe: pd.DataFrame):
    column_cleaner.run(dataframe)
    type_converter.run(dataframe)

# df = pd.read_csv("/home/jovyan/work/Autobahn/dataset/data_csv.csv")
# print(df['Sex'])
# print(df['Sex'].unique())
# all_in_one(df)
# print(df['Sex'])
# print(df['Sex'].unique())