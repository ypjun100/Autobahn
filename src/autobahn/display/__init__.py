import io
import uuid
import codecs
import functools
import pandas as pd
import ipywidgets as widgets

from autobahn.utils import Pipeline
from autobahn.preprocessing import scaler
import autobahn.preprocessing as preprocessing
from autobahn.utils.dataset import get_dataset_statistics
from autobahn.preprocessing import type_converter, missing_value, category_encoder
from autobahn.modeling.regression import Regression
from autobahn.modeling.classification import Classification

class Tabs:
    NUMBER_OF_TABS = 6
    TITLES = ["Dataset Upload", "Verification", "Preprocessing", "Final Dataset", "Modeling", "Analysis"]

    def __init__(self):
        self.tab = widgets.Tab()
        self.filename = ''                # Uploaded filename
        self.dataframe = None             # Raw dataset frame
        self.dataset = None               # Preprocessed dataset
        self.dependent_col = ""           # Name of dependent variable
        self.is_echo = False              # Prevent echoing dropdown menu
        self.pipeline = None              # Data preprocessing pipeline
        self.scaling_method = {}          # Scaling method of each variable (Default: False)
        self.model = None                 # Final model
        self.shap_model = None            # Final shap model
        self.clf = Classification()       # For building model and predicting values
        self.reg = Regression()           # For getting shap value
        
        # Prevent omitting dataframe output
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)

    def get_title(self, index):
        return Tabs.TITLES[index]
    
    def get_view(self, index):
        if index == 0:
            return self.get_upload_view()
        elif index == 1:
            return self.get_dataset_verfication_view()
        elif index == 2:
            return self.get_data_preprocessing_view()
        elif index == 3:
            return self.get_final_dataset_view()
        elif index == 4:
            return self.get_auto_modeling_view()
        elif index == 5:
            return self.get_model_analysis_view()


    ##################
    # 0 - Upload File
    ##################
    def upload_file(self, change):
        content = ''
        if len(change['new']) == 1:
            self.filename = change['new'][0]['name']
            content = change['new'][0]['content']
        else:
            file = change['new'][list(change['new'].keys())[0]]
            self.filename = file['metadata']['name']
            content = file['content']
        self.dataframe = pd.read_csv(io.StringIO(codecs.decode(content)), na_values=missing_value.MISSING_VALUE_SYMBOL)
        preprocessing.pre_all_in_one(self.dataframe) # Data preprocessing
        for col in self.dataframe.columns:
            self.scaling_method[col] = 'False'
        self.update_dataset_verification_view()
        self.update_data_preprocessing_view()
        self.tab.selected_index = 1
        
    def get_upload_view(self): # 0
        uploader = widgets.FileUpload(accept='*.csv', multiple=False)
        uploader.observe(self.upload_file, names="value")
        return uploader
    

    ##################
    # 1 - Data Verification
    ##################
    def get_dataset_verfication_view(self):
        self.data_verfication_vbox = widgets.VBox([])
        return self.data_verfication_vbox
    
    def update_dataset_verification_view(self):
        children = []
        children.append(widgets.HTML(value="<h4>Number of rows : " + str(len(self.dataframe)) + "</h4>"))
        children.append(widgets.HTML(value="<h4>Input Dataset</h4>"))
        children.append(widgets.HTML(value=self.dataframe.head()._repr_html_()))
        children.append(widgets.HTML(value="<h4>Statistics of Input Dataset</h4>"))
        children.append(widgets.HTML(value=get_dataset_statistics(self.dataframe).style.hide()._repr_html_()))
        self.data_verfication_vbox.children = children
    
    
    ##################
    # 2 - Pre - Data Preprocessing
    ##################
    def get_data_preprocessing_view(self):
        self.data_preprocessing_vbox = widgets.VBox([])
        self.data_preprocessing_loading = widgets.IntProgress(value=0, min=0, max=10, description="Waiting", bar_style="info")
        return self.data_preprocessing_vbox
    
    def on_dependent_change(self, change):
        self.dependent_col = change['new']

    def on_type_change(self, col, change):
        if self.is_echo:
            self.is_echo = False
            return
        
        if type_converter.convert_column_type(self.dataframe, col, change['new']):
            self.update_data_preprocessing_view()
            self.update_dataset_verification_view()
        else:
            self.is_echo = True
            change['owner'].value = change['old']
        

    def on_col_delete(self, col, _):
        self.dataframe.drop(col, axis = 1, inplace=True)
        if col in self.scaling_method:
            self.scaling_method[col] = 'False'
        self.update_data_preprocessing_view()
        self.update_dataset_verification_view()

    def on_scaling_method_change(self, col, change):
        self.scaling_method[col] = change['new']
    
    def on_preprocessing_apply(self, _):
        # Copy dataframe to dataset
        self.dataset = self.dataframe.copy()

        self.data_preprocessing_loading.description = "Processing..."
        missing_value.run(self.dataset, self.dependent_col)
        self.data_preprocessing_loading.value += 5

        # Initalize pipeline
        self.pipeline = Pipeline(self.dataset, self.dependent_col)

        # Dataset encoding
        category_encoder.run(self.dataset, self.dependent_col, self.pipeline)
        self.data_preprocessing_loading.value += 3

        # Dataset scaling
        for column in self.scaling_method.keys():
            if self.scaling_method[column] == 'Normalize':
                scaler.normalize(self.dataset, [column], self.pipeline)
            elif self.scaling_method[column] == 'Standardize':
                scaler.standardize(self.dataset, [column], self.pipeline)
        self.data_preprocessing_loading.value += 2

        self.update_final_dataset_view()
        self.tab.selected_index = 3
        self.data_preprocessing_loading.value = 0
        self.data_preprocessing_loading.description = ""
        
    def update_data_preprocessing_view(self):
        children = []
        dropdown_dependent = widgets.Dropdown(options=self.dataframe.columns.to_list(), description="Dependent Variable")
        dropdown_dependent.observe(self.on_dependent_change, names='value')
        self.dependent_col = dropdown_dependent.value
        children.append(dropdown_dependent)
        children.append(widgets.HTML(value="<hr/>"))
        children.append(widgets.HBox([widgets.Dropdown(options=['Type'], description="Feature name", disabled=True),
                                      widgets.Dropdown(options=['Scaling Method'], disabled=True)]))
        for col in self.dataframe.columns:
            dropdown_type = widgets.Dropdown(options=['Categorical', 'Numeric'], value='Categorical' if str(self.dataframe[col].dtypes) in ['bool', 'category'] else 'Numeric', description=col)
            dropdown_type.observe(functools.partial(self.on_type_change, col), names='value')
            dropdown_scaling_method = widgets.Dropdown(options=['False', 'Normalize', 'Standardize'], value=self.scaling_method[col], disabled=False if dropdown_type.value == 'Numeric' else True)
            dropdown_scaling_method.observe(functools.partial(self.on_scaling_method_change, col), names='value')
            button_col_delete = widgets.Button(description="Delete")
            button_col_delete.on_click(functools.partial(self.on_col_delete, col))
            children.append(widgets.HBox([dropdown_type, dropdown_scaling_method, button_col_delete]))
        children.append(widgets.HTML(value="<hr/>"))
        apply_button = widgets.Button(description="Apply")
        apply_button.on_click(self.on_preprocessing_apply)
        children.append(widgets.HBox([apply_button, self.data_preprocessing_loading]))
        self.data_preprocessing_vbox.children = children
    

    ##################
    # 3 - Final Dataset
    ##################
    def get_final_dataset_view(self):
        self.final_dataset_vbox = widgets.VBox([])
        return self.final_dataset_vbox
    
    def on_start_modeling(self, _):
        self.update_auto_modeling_view()
        self.tab.selected_index = 4
        self.start_modeling()
    
    def update_final_dataset_view(self):
        children = []
        children.append(widgets.HTML(value="<h4>Number of rows : " + str(len(self.dataset)) + "</h4>"))
        children.append(widgets.HTML(value="<h4>Dependent Variable : " + self.dependent_col))
        children.append(widgets.HTML(value="<h4>Final Dataset</h4>"))
        children.append(widgets.HTML(value=self.dataset.head()._repr_html_()))
        children.append(widgets.HTML(value="<h4>Statistics of Final Dataset</h4>"))
        children.append(widgets.HTML(value=get_dataset_statistics(self.dataset).style.hide()._repr_html_()))
        children.append(widgets.HTML(value="<hr/>"))
        start_modeling_button = widgets.Button(description="Start Modeling")
        start_modeling_button.on_click(self.on_start_modeling)
        children.append(start_modeling_button)
        self.final_dataset_vbox.children = children


    ##################
    # 4 - Auto Modeling
    ##################
    def get_auto_modeling_view(self):
        self.auto_modeling_vbox = widgets.VBox([])
        return self.auto_modeling_vbox
    
    def update_auto_modeling_view(self):
        children = []
        self.auto_modeling_output = widgets.Output()
        children.append(self.auto_modeling_output)
        self.score_table = widgets.HTML()
        children.append(self.score_table)
        children.append(widgets.HTML(value="<hr/>"))
        save_model_button = widgets.Button(description="Save Model")
        save_model_button.on_click(self.on_save_model)
        children.append(save_model_button)
        self.auto_modeling_vbox.children = children

    def start_modeling(self):
        self.auto_modeling_output.clear_output()
        with self.auto_modeling_output:
            # Training classification model
            result = self.clf.train(self.dataset, self.dependent_col)
            self.model = result['model']
            
            # Training regression model
            self.shap_model = self.reg.train(self.dataset, self.dependent_col, True)

            self.score_table.value = result['score_table']._repr_html_()

        self.update_model_analysis_view()
    
    def on_save_model(self, _):
        random_uuid = str(uuid.uuid4()).split('-')[0]
        if self.model != None and self.shap_model != None and self.pipeline != None:
            self.clf.save(self.model, "model-" + random_uuid)
            self.reg.save(self.shap_model, 'shap-model-' + random_uuid)
            self.pipeline.save('pipeline-' + random_uuid)
            self.dataset.to_pickle('dataset-' + random_uuid + '.pkl')
            print('Complete to save -', random_uuid)
        else:
            print('Invalid model or pipeline')
    
    
    ##################
    # 5 - Model Analysis
    ##################
    def get_model_analysis_view(self):
        self.model_analysis_vbox = widgets.VBox([])
        return self.model_analysis_vbox
    
    def on_plot_change(self, plot, _):
        self.model_analysis_output.clear_output()
        with self.model_analysis_output:
            if plot == 'shap':
                self.reg.plot(self.shap_model, plot)
            elif plot in ['confusion_matrix', 'feature']:
                self.clf.plot(self.model, plot)

    def update_model_analysis_view(self):
        children = []
        shap_button = widgets.Button(description="SHAP")
        shap_button.on_click(functools.partial(self.on_plot_change, 'shap'))
        cm_button = widgets.Button(description="Confusion Matrix")
        cm_button.on_click(functools.partial(self.on_plot_change, 'confusion_matrix'))
        fi_button = widgets.Button(description="Feature Importance")
        fi_button.on_click(functools.partial(self.on_plot_change, 'feature'))
        children.append(widgets.HBox([shap_button, cm_button, fi_button]))
        self.model_analysis_output = widgets.Output()
        children.append(self.model_analysis_output)
        self.model_analysis_vbox.children = children


def display():
    tabs = Tabs()
    tabs.tab.children = [tabs.get_view(i) for i in range(Tabs.NUMBER_OF_TABS)]
    tabs.tab.titles = tabs.TITLES

    return widgets.VBox([widgets.HTML(value="<h2>Autobahn: AUTOmation Brings All Human beiNgs.</h2>"), tabs.tab])