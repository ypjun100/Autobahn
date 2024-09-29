import pandas as pd
from pycaret.classification import *

def classification(dataset:pd.DataFrame, target: str) -> dict:
    # Setup for training
    print('Train parameters')
    setup(data=dataset, target=target, train_size=0.8, session_id=42)
    result = pull()
    print(result)

    # Train
    print('Training model...')
    best_model = compare_models(sort='Accuracy', n_select=1, fold=2)
    finalize_model(best_model)
    result = pull()

    # Determine best shap model
    print('Determining best shap model...')
    best_shap_model = compare_models(sort='Accuracy', n_select=1, fold=2, include=['dt', 'lightgbm', 'et', 'rf'])
    finalize_model(best_shap_model)
    return {'model': best_model, 'shap_model': best_shap_model, 'result_table': result}

def save(model, filename: str):
    save_model(model, filename, verbose=False)

def load(filename: str):
    return load_model(filename, verbose=False)

def predict(model, data):
    return predict_model(model, data, verbose=False)

def plot(model, plot:str = ''):
    if plot == 'shap':
        interpret_model(model)
    elif plot in ['confusion_matrix', 'feature']:
        plot_model(model, plot)