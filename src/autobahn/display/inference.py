import os
import pandas as pd
import ipywidgets as widgets

from autobahn.utils import Pipeline
from autobahn.explainer import Explainer
from autobahn.modeling.regression import Regression
from autobahn.modeling.classification import Classification

class Tabs:
    NUMBER_OF_TABS = 2
    TITLES = ['모델 선택', '예측']

    def __init__(self):
        self.tab = widgets.Tab()
        self.clf = Classification()
        self.reg = Regression()
        self.pipeline = {}
        self.model = None
        self.shap_model = None
        self.dataset = None
        self.explainer = None

    def get_title(self, index):
        return Tabs.TITLES[index]
    
    def get_view(self, index):
        if index == 0:
            return self.get_model_selecting_view()
        elif index == 1:
            return self.get_prediction_view()
    

    ###################
    # 0 - Model Selecting
    ###################
    def get_model_selecting_view(self):
        self.model_selecting_text = widgets.Text(value='', description='Model ID :')
        submit_button = widgets.Button(description="Submit")
        submit_button.on_click(self.on_model_submit)
        vbox = widgets.VBox([
            self.model_selecting_text,
            widgets.HTML(value="<hr/>"),
            submit_button
        ])
        return vbox
    
    def on_model_submit(self, _):
        if self.model_selecting_text.value != '':
            self.pipeline = Pipeline.open('pipeline-' + self.model_selecting_text.value)
            self.model = self.clf.load('model-' + self.model_selecting_text.value)
            self.shap_model = self.reg.load('shap-model-' + self.model_selecting_text.value)
            self.dataset = pd.read_pickle('dataset-' + self.model_selecting_text.value + '.pkl')
            os.environ["HF_TOKEN"] = ""
            self.explainer = Explainer(llm_model="llama")
            self.get_prediction_questions()
            self.update_prediction_view()
            self.tab.selected_index = 1

    
    ###################
    # 1 - Prediction
    ###################
    def get_prediction_view(self):
        self.prediction_vbox = widgets.VBox([])
        return self.prediction_vbox
    
    def get_prediction_questions(self):
        self.prediction_question_widgets = {}
        for key in self.pipeline:
            if self.pipeline[key]['type'] == 'numeric':
                text = widgets.Text(description=key + " :")
                self.prediction_question_widgets[key] = text
            elif self.pipeline[key]['type'] == 'category':
                dropdown = widgets.Dropdown(options=self.pipeline[key]['category'], description=key + " :")
                self.prediction_question_widgets[key] = dropdown

    def explain(self, _prediction_df):
        # Append user input & prediction to dataset
        dependent_col = [col for col in self.dataset.columns if col not in self.pipeline.keys()][0]
        prediction_df =  _prediction_df.rename(columns={ 'prediction_label': dependent_col })

        if ('prediction_score' in prediction_df.columns):
            prediction_df.drop(['prediction_score'], axis=1, inplace=True)
        combined_dataset = pd.concat([self.dataset, prediction_df], ignore_index=True)

        # Waterfall plot
        with self.result_plot_output:
            self.explainer.plot_waterfall(self.shap_model, combined_dataset, dependent_col)

        # Get explaination
        explaination_result = self.explainer.explain(self.shap_model, combined_dataset, dependent_col)
        self.result_explaination.value = f'<p style="font-weight: bold">Explaination</p><p style="max-width: 100%;">{explaination_result}</p>'

    def on_predict(self, _):
        self.result_plot_output.clear_output()
        input = {}
        for key in self.prediction_question_widgets:
            input[key] = self.prediction_question_widgets[key].value
            if self.pipeline[key]['type'] == 'numeric':
                if self.pipeline[key]['scaling'] == 'Normalize':
                    input[key] = self.pipeline[key]['normalizer'].transform(float(self.prediction_question_widgets[key].value))
                elif self.pipeline[key]['scaling'] == 'Standardize':
                    input[key] = self.pipeline[key]['standardizer'].transform(float(self.prediction_question_widgets[key].value))
            elif self.pipeline[key]['type'] == 'category':
                if self.pipeline[key]['encoding'] == 'True':
                    input[key] = self.pipeline[key]['encoder'].transform(self.prediction_question_widgets[key].value)
        user_input_df = pd.DataFrame([input])
        result_df = self.clf.predict(self.model, user_input_df)
        self.result_prediction.value = '<p style="font-weight: bold">Prediction : ' + str(result_df['prediction_label'][0]) + '</p>'
        self.explain(result_df)

    def update_prediction_view(self):
        children = []
        for key in self.prediction_question_widgets:
            children.append(self.prediction_question_widgets[key])
        children.append(widgets.HTML(value="<hr/>"))
        predict_button = widgets.Button(description="Predict")
        predict_button.on_click(self.on_predict)
        children.append(predict_button)
        children.append(widgets.HTML(value="<hr/>"))
        self.result_plot_output = widgets.Output()
        self.result_prediction = widgets.HTML('')
        self.result_explaination = widgets.HTML('')
        children.extend([self.result_plot_output, self.result_prediction, self.result_explaination])
        self.prediction_vbox.children = children


def display():
    tabs = Tabs()
    tabs.tab.children = [tabs.get_view(i) for i in range(Tabs.NUMBER_OF_TABS)]
    tabs.tab.titles = tabs.TITLES

    return widgets.VBox([widgets.HTML(value="<h2>Autobahn: AUTOmation Brings All Human beiNgs.</h2>"), tabs.tab])