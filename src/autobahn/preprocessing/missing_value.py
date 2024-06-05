import gc
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

MISSING_VALUE_SYMBOL = ['-', '--', 'na', 'NAN', 'n/a', '?']
IMPUTATION_STRATEGIES = ['mean', 'median', 'most_frequent', 'knn']


# Find columns have missing value
def detect(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.columns[dataframe.isnull().any()].tolist()


# Replace missing value in series to NaN
def replace_series(series: pd.Series, to = None) -> pd.Series:
    tmp = series.copy()
    for i in range(len(tmp)):
        if pd.isna(tmp[i]) or tmp[i] in MISSING_VALUE_SYMBOL:
            tmp[i] = to
    return tmp


# If ratio of missing values in column is over 90%, remove column
def clean(dataframe: pd.DataFrame):
    for column in detect(dataframe):
        if int(len(dataframe) * 0.9) <= sum(replace_series(dataframe[column]).isna()):
            dataframe.drop(column, axis=1, inplace=True)


# Get imputation strategy which has highest score
def evaluate_preferred_strategy(dataframe: pd.DataFrame, column: str, target: str) -> str:

    # Create new dataframe dropped missing values
    _dataframe = dataframe.dropna(subset=[elem for elem in detect(dataframe) if elem != column])
    numeric_columns = dataframe.select_dtypes('number').columns.to_list()
    preferred_strategy, preferred_strategy_score = 'most_frequent', 0

    # Execute label encoding for categorical column
    le = LabelEncoder()
    for col in _dataframe.columns:
        if col in numeric_columns:
            continue
        _dataframe[col] = le.fit_transform(_dataframe[col])

    for strategy in IMPUTATION_STRATEGIES:
        __dataframe = _dataframe.copy()
        score = 0.0
        
        # Only numeric colum can be applied all types of methods
        if column in numeric_columns:
            if strategy != 'knn':
                imputer = SimpleImputer(missing_values=pd.NA, strategy=strategy)
                __dataframe[column] = imputer.fit_transform(__dataframe[column].to_numpy().reshape(-1, 1)).reshape(-1)
            else:
                imputer = KNNImputer(n_neighbors=5)
                __dataframe[column] = imputer.fit_transform(__dataframe[column].to_numpy().reshape(-1, 1)).reshape(-1)

            # Evaluate
            X_train, X_test, y_train, y_test = train_test_split(__dataframe.drop(target, axis=1), __dataframe[target],
                                                                test_size=0.3, random_state=42)
            clf = GaussianNB().fit(X_train, y_train)
            dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
            score = (clf.score(X_test, y_test) + dt.score(X_test, y_test)) / 2.0
            
            del X_train, X_test, y_train, y_test, clf, dt
        else: # Categorical colum can be applied only mode and knn
            if strategy in ['most_frequent', 'knn']:
                if strategy != 'knn':
                    imputer = SimpleImputer(missing_values=pd.NA, strategy=strategy)
                    __dataframe[column] = imputer.fit_transform(__dataframe[column].to_numpy().reshape(-1, 1)).reshape(-1)
                else:
                    imputer = KNNImputer(n_neighbors=5)
                    __dataframe[column] = imputer.fit_transform(__dataframe[column].to_numpy().reshape(-1, 1)).reshape(-1)
                
                # Evaluate
                score = 0.0
                X_train, X_test, y_train, y_test = train_test_split(__dataframe.drop(target, axis=1), __dataframe[target],
                                                                    test_size=0.3, random_state=42)
                clf = GaussianNB().fit(X_train, y_train)
                dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
                score = (clf.score(X_test, y_test) + dt.score(X_test, y_test)) / 2.0

                del X_train, X_test, y_train, y_test, clf, dt

        if preferred_strategy_score < score:
            preferred_strategy = strategy
            preferred_strategy_score = score

        del __dataframe, score
    
    gc.collect()
    return preferred_strategy


# Main runner
def run(dataframe: pd.DataFrame, target: str):
    clean(dataframe)
    for column in detect(dataframe):
        preferred_method = evaluate_preferred_strategy(dataframe, column, target)
        print(column, preferred_method)