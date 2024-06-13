import pandas as pd
import ipywidgets as widgets

from autobahn.modeling import load, predict
from autobahn.utils import Pipeline

class Tabs:
    NUMBER_OF_TABS = 2
    TITLES = ['모델 선택', '예측']

    def __init__(self):
        self.tab = widgets.Tab()
        self.pipeline = {}
        self.model = None

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
        hbox = widgets.HBox([
            self.model_selecting_text,
            submit_button
        ])
        return hbox
    
    def on_model_submit(self, _):
        if self.model_selecting_text.value != '':
            self.pipeline = Pipeline.open('pipeline-' + self.model_selecting_text.value)
            self.model = load('model-' + self.model_selecting_text.value)
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

    def on_predict(self, _):
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
        input = pd.DataFrame([input])
        result = predict(self.model, input)
        self.result_prediction.value = '<p style="font-weight: bold">Prediction : ' + str(result['prediction_label'][0]) + '</p>'
        self.result_score.value = '<p style="font-weight: bold">Score : ' + str(result['prediction_score'][0]) + '</p>'

    def update_prediction_view(self):
        children = []
        for key in self.prediction_question_widgets:
            children.append(self.prediction_question_widgets[key])
        children.append(widgets.HTML(value="<hr/>"))
        predict_button = widgets.Button(description="Predict")
        predict_button.on_click(self.on_predict)
        children.append(predict_button)
        children.append(widgets.HTML(value="<hr/>"))
        self.result_prediction = widgets.HTML('')
        self.result_score = widgets.HTML('')
        children.extend([self.result_prediction, self.result_score])
        self.prediction_vbox.children = children


def display():
    tabs = Tabs()
    tabs.tab.children = [tabs.get_view(i) for i in range(Tabs.NUMBER_OF_TABS)]
    tabs.tab.titles = tabs.TITLES

    return widgets.VBox([widgets.HTML(value="<h2>Autobahn: AUTOmation Brings All Human beiNgs.</h2>"), tabs.tab])