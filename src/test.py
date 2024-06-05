import io
import codecs
import functools
import numpy as np
import pandas as pd
import ipywidgets as widgets

class MissingValueProcessor:
    # Remove rows of missing value
    def remove(dataframe, target_columns):
        _df = dataframe.copy()
        for target_column in target_columns:
            _df.dropna(subset=[target_column], inplace=True)
        return _df
    
    # Fill values
    def fill(dataframe, target_columns):
        _df = dataframe.copy()
        for target_column in target_columns:
            subset_without_nan = _df.dropna(subset=[target_column])
            _df[target_column].fillna(np.random.choice(subset_without_nan[target_column], size=1)[0], inplace=True)
        return _df


class Tabs:
    NUMBER_OF_TABS = 3
    TITLES = ["데이터셋 업로드", "데이터셋 확인", "데이터 전처리"]

    def __init__(self):
        self.tab = widgets.Tab()
        self.dataframe = None          # 데이터 변수
        self.dependent_col = ""        # 종속변수 명
        self.is_echo = False           # Dropdown 에코 방지
        self.is_auto_preprocessing = {}    # 각 변수별 auto transformation 여부 {변수명:True/False...}  (기본값 : True)

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
        self.dataframe = pd.read_csv(io.StringIO(codecs.decode(content)))
        for col in self.dataframe.columns:
            self.is_auto_preprocessing[col] = True
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
    
    def get_dataset_statistics(self):
        statistics = pd.DataFrame({'Feature Name': self.dataframe.columns.to_list(),
                           'Type': ['Categorical' if self.dataframe[col].dtypes == 'object' else 'Numeric' for col in self.dataframe.columns],
                           'Unique': [str(len(self.dataframe[col].unique())) if self.dataframe[col].dtypes == 'object' else '-' for col in self.dataframe.columns],
                           'Missing values': self.dataframe.isnull().sum().to_list(),
                           'Mean': [str(round(self.dataframe[col].mean(), 5)) if self.dataframe[col].dtypes != 'object' else '-' for col in self.dataframe.columns]})
        return statistics.style.hide()._repr_html_()
    
    def update_dataset_verification_view(self):
        children = []
        children.append(widgets.HTML(value="<h4>총 데이터 수 : " + str(len(self.dataframe)) + "개</h4>"))
        children.append(widgets.HTML(value="<h4>입력 데이터셋</h4>"))
        children.append(widgets.HTML(value=self.dataframe.head()._repr_html_()))
        children.append(widgets.HTML(value="<h4>입력 데이터셋 통계치</h4>"))
        children.append(widgets.HTML(value=self.get_dataset_statistics()))
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
        
        try:
            if change['new'] == 'Categorical': # Numeric -> Categorical
                self.dataframe[col] = self.dataframe[col].astype(object)
            else: # Categorical -> Numeric
                self.dataframe[col] = self.dataframe[col].astype(int)
            self.update_dataset_verification_view()
        except:
            self.is_echo = True
            change['owner'].value = change['old']
            print('변환할 수 없는 변수입니다. -', col)

    def on_col_delete(self, col, _):
        self.dataframe.drop(col, axis = 1, inplace=True)
        self.update_data_preprocessing_view()
        self.update_dataset_verification_view()

    def on_auto_preprocessing_change(self, col, change):
        self.is_auto_preprocessing[col] = False if change['new'] == 'None' else True
    
    def on_preprocessing_apply(self, _):
        self.data_preprocessing_loading.description = "Processing..."

        # 어떤 컬럼이 자동 전처리에 해당하는지 확인
        auto_transform_cols, non_auto_transform_cols = [], []
        for key in self.is_auto_preprocessing.keys():
            if self.is_auto_preprocessing[key]:
                auto_transform_cols.append(key)
            else:
                non_auto_transform_cols.append(key)

        # auto transform이 false라면, 결측치 제거
        # auto transform이 true라면, 결측치 랜덤 값 및 범주로 채워넣기
        self.dataframe = MissingValueProcessor.remove(self.dataframe, non_auto_transform_cols) # 결측치 제거
        self.data_preprocessing_loading.value += 2
        self.dataframe = MissingValueProcessor.fill(self.dataframe, auto_transform_cols) # 랜덤 값/범주 채워넣기
        self.data_preprocessing_loading.value += 2
        




    def update_data_preprocessing_view(self):
        children = []
        dropdown_dependent = widgets.Dropdown(options=self.dataframe.columns.to_list(), description="종속 변수 설정")
        dropdown_dependent.observe(self.on_dependent_change, names='value')
        children.append(dropdown_dependent)
        children.append(widgets.HTML(value="<hr/>"))
        children.append(widgets.HBox([widgets.Dropdown(options=['Type'], description="Feature name", disabled=True),
                                      widgets.Dropdown(options=['Auto Preprocessing'], disabled=True)]))
        for col in self.dataframe.columns:
            dropdown_type = widgets.Dropdown(options=['Categorical', 'Numeric'], value='Categorical' if self.dataframe[col].dtypes == 'object' else 'Numeric', description=col)
            dropdown_type.observe(functools.partial(self.on_type_change, col), names='value')
            dropdown_transform = widgets.Dropdown(options=['Auto', 'None'], value='Auto' if self.is_auto_preprocessing[col] == True else 'None')
            dropdown_transform.observe(functools.partial(self.on_auto_preprocessing_change, col), names='value')
            button_col_delete = widgets.Button(description="Delete")
            button_col_delete.on_click(functools.partial(self.on_col_delete, col))
            children.append(widgets.HBox([dropdown_type, dropdown_transform, button_col_delete]))
        children.append(widgets.HTML(value="<hr/>"))
        apply_button = widgets.Button(description="Apply")
        apply_button.on_click(self.on_preprocessing_apply)
        children.append(widgets.HBox([apply_button, self.data_preprocessing_loading]))
        self.data_preprocessing_vbox.children = children