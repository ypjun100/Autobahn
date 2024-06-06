import gc
from datetime import date
import warnings
import pandas as pd

# import missing_value
from autobahn.preprocessing import missing_value

warnings.filterwarnings(action='ignore')

DEFAULT_DATETIME = '2000/10/19 01:01:01'


# Split column to year, month, day, hour, minute, second
def split_datetime(dataframe: pd.DataFrame, column: str):
    try:
        year, month, day = [], [], []
        hour, minute, second = [], [], []
        for i in range(len(dataframe)):
            if str(dataframe.loc[i, column]) == 'NaT':
                year.append(None)
                month.append(None)
                day.append(None)
                hour.append(None)
                minute.append(None)
                second.append(None)
                continue

            year.append(int(dataframe.loc[i, column].strftime("%Y")))
            month.append(int(dataframe.loc[i, column].strftime("%m")))
            day.append(int(dataframe.loc[i, column].strftime("%d")))

            hour.append(int(dataframe.loc[i, column].strftime("%H")))
            minute.append(int(dataframe.loc[i, column].strftime("%M")))
            second.append(int(dataframe.loc[i, column].strftime("%S")))
    except:
        pass

    dataframe[column + "_year"] = year
    dataframe[column + "_month"] = month
    dataframe[column + "_day"] = day

    dataframe[column + "_hour"] = hour
    dataframe[column + "_minute"] = minute
    dataframe[column + "_second"] = second

    # Convert to int
    dataframe[column + "_year"] = dataframe[column + "_year"].astype("Int64")
    dataframe[column + "_month"] = dataframe[column + "_month"].astype("Int64")
    dataframe[column + "_day"] = dataframe[column + "_day"].astype("Int64")

    dataframe[column + "_hour"] = dataframe[column + "_hour"].astype("Int64")
    dataframe[column + "_minute"] = dataframe[column + "_minute"].astype("Int64")
    dataframe[column + "_second"] = dataframe[column + "_second"].astype("Int64")

    del year, month, day, hour, minute, second
    gc.collect()



def run(dataframe: pd.DataFrame):

    # Convert numeric columns
    for column in dataframe.select_dtypes('number').columns:
        # If number of unique values in numeric is two, convert to bool
        if len(dataframe[column].dropna(axis=0).unique()) == 2:
            unique_values = dataframe[column].dropna(axis=0).unique()
            dataframe[column].replace({unique_values[0]: 0, unique_values[1]: 1}, inplace=True)
            dataframe[column] = dataframe[column].astype(bool)
            del unique_values
            continue

    # Convert object(categorical) columns
    # Each column is attempted to convert into some data type like bool, date, category.
    for column in dataframe.select_dtypes('object').columns:    

        # convert to datetime
        try:
            dataframe[column] = pd.to_datetime(missing_value.replace_series(dataframe[column], DEFAULT_DATETIME), format='mixed')
            
            # Replace missing data to NaT
            for i in range(len(dataframe)):
                if dataframe.loc[i, column].strftime('%Y/%m/%d %H:%M:%S') == DEFAULT_DATETIME:
                    dataframe.loc[i, column] = pd.NaT

            # Split column
            split_datetime(dataframe, column)

            dataframe.drop(column, axis=1, inplace=True)
            continue
        except:
            pass

        # convert to bool
        if len(dataframe[column].dropna(axis=0).unique()) == 2:
            unique_values = dataframe[column].dropna(axis=0).unique()
            dataframe[column].replace({unique_values[0]: 0, unique_values[1]: 1}, inplace=True)
            dataframe[column] = dataframe[column].astype(bool)
            del unique_values
            continue

        # convert to category
        if len(dataframe[column].unique()) < len(dataframe[column]):
            dataframe[column] = dataframe[column].astype("category")
            continue

        # If column couldn't convert any of above steps, delete column.
        dataframe.drop(column, axis=1, inplace=True)
    
    gc.collect() # for memory deallocate


# Convert numeric to category
def numeric_to_category(dataframe: pd.DataFrame, column: str):    
    if len(dataframe[column].dropna(axis=0).unique()) == 2:       # convert to bool
        unique_values = dataframe[column].dropna(axis=0).unique()
        dataframe[column].replace({unique_values[0]: 0, unique_values[1]: 1}, inplace=True)
        dataframe[column] = dataframe[column].astype(bool)
        del unique_values
    elif len(dataframe[column].unique()) < len(dataframe[column]): # convert to category
        dataframe[column] = dataframe[column].astype("category")
        
    gc.collect() # for memory deallocate


# Convert category to numeric
def category_to_numeric(dataframe: pd.DataFrame, column: str):
    dataframe[column] = dataframe[column].astype('Float64')
    if (dataframe[column].fillna(-9999) % 1  == 0).all():
        dataframe[column] = dataframe[column].astype('Int64')
    else:
        dataframe[column] = dataframe[column].astype('Float64')


# If converting process succeed, return True or False
def convert_column_type(dataframe: pd.DataFrame, column: str, to: str) -> bool:
    try:
        # Numeric -> Categorical
        if to == 'Categorical':
            numeric_to_category(dataframe, column)
        # Categorical -> Numeric
        else: 
            category_to_numeric(dataframe, column)
        return True
    except Exception as e:
        print(e)
    return False

# df = pd.read_csv("/home/jovyan/work/Autobahn/dataset/data_csv.csv")
# df = pd.read_csv("/home/jovyan/work/Autobahn/dataset/mxmh_survey_results.csv")
# df = pd.read_csv("/home/jovyan/work/Autobahn/dataset/cms_hospital_patient_satisfaction_2020.csv")
# df = pd.read_csv("/home/jovyan/work/Autobahn/dataset/NHANESI_subset.csv")
# run(df)
# numeric_to_category(df, 'Age_Years')
# category_to_numeric(df, 'Age_Years')
# print(df.dtypes)
# print(df.head())