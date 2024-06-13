import gc
import pandas as pd
from pycaret.classification import *

def classification(dataset:pd.DataFrame, target: str) -> dict:
    setup(data=dataset, target=target, train_size=0.8, session_id=42)
    result = pull()
    print(result)
    best_model = compare_models(sort='Accuracy', n_select=1, fold=2)
    result = pull()
    return {'model': best_model, 'result_table': result}

def save(model, filename: str):
    save_model(model, filename, verbose=False)

def load(filename: str):
    return load_model(filename, verbose=False)

def predict(model, data):
    return predict_model(model, data, verbose=False)