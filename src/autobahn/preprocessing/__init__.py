import pandas as pd
from autobahn.utils.dataset import verify
from autobahn.preprocessing import missing_value
from autobahn.preprocessing import column_cleaner
from autobahn.preprocessing import type_converter
from autobahn.preprocessing import category_encoder


# Run pre steps in preprocessing
def pre_all_in_one(dataframe: pd.DataFrame):
    # Verify dataset
    verification_result = verify(dataframe)
    if not verification_result[0]:
        print(verification_result[1])
        return

    column_cleaner.run(dataframe)
    type_converter.run(dataframe)

# Run post steps in preprocessing
def post_all_in_one(dataframe: pd.DataFrame, target: str):
    # Verify dataset
    verification_result = verify(dataframe, target)
    if not verification_result[0]:
        print(verification_result[1])
        return
    
    missing_value.run(dataframe, target)
    category_encoder.run(dataframe)