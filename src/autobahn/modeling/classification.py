import pandas as pd
from pycaret.classification import ClassificationExperiment

class Classification:
    def __init__(self):
        self.exp = ClassificationExperiment()
    
    def train(self, dataset: pd.DataFrame, target: str):
        # Setting up
        self.exp.setup(data=dataset, target=target, train_size=0.8, session_id=42)
        print(self.exp.pull())

        # Training
        print('Training model...')
        best_model = self.exp.compare_models(sort='Accuracy', n_select=1, fold=2)
        score_table = self.exp.pull()

        return {
            'model': best_model,
            'score_table': score_table
        }

    def save(self, model, filename: str):
        self.exp.save_model(model, filename, verbose=False)

    def load(self, filename: str):
        return self.exp.load_model(filename, verbose=False)

    def predict(self, model, data):
        return self.exp.predict_model(model, data, verbose=False)

    def plot(self, model, plot: str = ''):
        if plot in ['shap']:
            self.exp.interpret_model(model)
        elif plot in ['confusion_matrix', 'feature']:
            self.exp.plot_model(model, plot)