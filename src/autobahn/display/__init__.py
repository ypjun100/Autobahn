import io
import codecs
import functools
import numpy as np
import pandas as pd
import ipywidgets as widgets

import autobahn.preprocessing as preprocessing
from autobahn.preprocessing import type_converter, missing_value
from autobahn.utils.dataset import get_dataset_statistics

class Tabs:
    NUMBER_OF_TABS = 3
    TITLES = ["데이터셋 업로드", "데이터셋 확인", "데이터 전처리"]

    def __init__(self):
        self.tab = widgets.Tab()
        self.dataframe = None             # 데이터 변수
        self.dependent_col = ""           # 종속변수 명
        self.is_echo = False              # Dropdown 에코 방지
        self.column_scaling_method = {}   # 각 변수별 스케일링 방식 지정 (기본값: False)

    def get_title(self, index):
        return Tabs.TITLES[index]
    
    def get_view(self, index):
        if index == 0:
            return self.get_upload_view()
        elif index == 1:
            return self.get_dataset_verfication_view()
        elif index == 2:
            return self.get_data_preprocessing_view()


    ##################
    # 0 - Upload File
    ##################
    def upload_file(self, change):
        content = change['new'][0]['content']
        self.dataframe = pd.read_csv(io.StringIO(codecs.decode(content)), na_values=missing_value.MISSING_VALUE_SYMBOL)
        preprocessing.all_in_one(self.dataframe) # 데이터 전처리
        for col in self.dataframe.columns:
            self.column_scaling_method[col] = 'False'
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
        children.append(widgets.HTML(value="<h4>총 데이터 수 : " + str(len(self.dataframe)) + "개</h4>"))
        children.append(widgets.HTML(value="<h4>입력 데이터셋</h4>"))
        children.append(widgets.HTML(value=self.dataframe.head()._repr_html_()))
        children.append(widgets.HTML(value="<h4>입력 데이터셋 통계치</h4>"))
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
            self.update_dataset_verification_view()
        else:
            self.is_echo = True
            change['owner'].value = change['old']
        

    def on_col_delete(self, col, _):
        self.dataframe.drop(col, axis = 1, inplace=True)
        self.update_data_preprocessing_view()
        self.update_dataset_verification_view()

    def on_auto_preprocessing_change(self, col, change):
        self.column_scaling_method[col] = change['new']
    
    def on_preprocessing_apply(self, _):
        self.data_preprocessing_loading.description = "Processing..."
        missing_value.run(self.dataframe, self.dependent_col)
        self.data_preprocessing_loading.value += 3
        self.update_dataset_verification_view()
        
        # 어떤 컬럼이 자동 전처리에 해당하는지 확인
        # for key in self.is_auto_preprocessing.keys():
        #     if self.is_auto_preprocessing[key]:
        #         auto_transform_cols.append(key)
        #     else:
        #         non_auto_transform_cols.append(key
        
    def update_data_preprocessing_view(self):
        children = []
        dropdown_dependent = widgets.Dropdown(options=self.dataframe.columns.to_list(), description="종속 변수 설정")
        dropdown_dependent.observe(self.on_dependent_change, names='value')
        children.append(dropdown_dependent)
        children.append(widgets.HTML(value="<hr/>"))
        children.append(widgets.HBox([widgets.Dropdown(options=['Type'], description="Feature name", disabled=True),
                                      widgets.Dropdown(options=['Scaling Method'], disabled=True)]))
        for col in self.dataframe.columns:
            dropdown_type = widgets.Dropdown(options=['Categorical', 'Numeric'], value='Categorical' if str(self.dataframe[col].dtypes) in ['bool', 'category'] else 'Numeric', description=col)
            dropdown_type.observe(functools.partial(self.on_type_change, col), names='value')
            dropdown_scaling_method = widgets.Dropdown(options=['False', 'Normalize', 'Standardize'], value=self.column_scaling_method[col])
            dropdown_scaling_method.observe(functools.partial(self.on_auto_preprocessing_change, col), names='value')
            button_col_delete = widgets.Button(description="Delete")
            button_col_delete.on_click(functools.partial(self.on_col_delete, col))
            children.append(widgets.HBox([dropdown_type, dropdown_scaling_method, button_col_delete]))
        children.append(widgets.HTML(value="<hr/>"))
        apply_button = widgets.Button(description="Apply")
        apply_button.on_click(self.on_preprocessing_apply)
        children.append(widgets.HBox([apply_button, self.data_preprocessing_loading]))
        self.data_preprocessing_vbox.children = children


def display():
    tabs = Tabs()
    tabs.tab.children = [tabs.get_view(i) for i in range(Tabs.NUMBER_OF_TABS)]
    tabs.tab.titles = tabs.TITLES

    return widgets.VBox([widgets.HTML(value="<h2>Autobahn: AUTOmation Brings All Human beiNgs.</h2>"), tabs.tab])