import pandas as pd

# 데이터셋 통계치 측정
# 변수명, 데이터 타입, 범주 수, 결측값 수, 평균
def get_dataset_statistics(dataframe):
    statistics = pd.DataFrame({'Feature Name': dataframe.columns.to_list(),
                        'Type': [dataframe[col].dtypes for col in dataframe.columns],
                        'Unique': [str(len(dataframe[col].unique())) if dataframe[col].dtypes in ['bool', 'category'] else '-' for col in dataframe.columns],
                        'Missing values': dataframe.isnull().sum().to_list(),
                        'Mean': [str(round(dataframe[col].mean(), 5)) if dataframe[col].dtypes not in ['bool', 'category'] else '-' for col in dataframe.columns]})
    return statistics