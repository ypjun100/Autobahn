import pandas as pd
from pycaret.regression import RegressionExperiment

class Regression:
    def __init__(self):
        self.exp = RegressionExperiment()
    
    def train(self, _dataset: pd.DataFrame, target: str, enable_variable_encoding: bool = False):
        # Deep copy from original dataframe
        dataset = _dataset.copy()

        # Variable encoding (Categorical -> Numeric)
        if (enable_variable_encoding):
            for column in dataset.columns:
                if (dataset[column].dtypes == 'category'):
                    try:
                        dataset[column] = dataset[column].astype('float')
                    except:
                        dataset[column] = dataset[column].astype('category').cat.codes

        # Setting up
        self.exp.setup(data=dataset, target=target, train_size=0.8, session_id=42, verbose=False)

        # Training
        print('Training model...')
        best_model = self.exp.compare_models(sort='MAE',
                                             n_select=1,
                                             fold=2,
                                             include=['dt', 'et', 'rf', 'lightgbm'])

        return best_model

    def save(self, model, filename: str):
        self.exp.save_model(model, filename, verbose=False)

    def load(self, filename: str):
        return self.exp.load_model(filename, verbose=False)

    def plot(self, model, plot: str = ''):
        if plot in ['shap']:
            self.exp.interpret_model(model)