import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import csv
from collections import OrderedDict, Counter
import re
from sklearn.linear_model import LinearRegression
import datetime
import seaborn as sns


def csv_to_json(file_path):
    CSV_PATH = file_path
    csv_file = csv.DictReader(open(CSV_PATH, 'r', encoding='UTF8'))
    colnames = list(pd.read_csv(CSV_PATH, low_memory=False).columns)

    json_dict = {}
    json_dict['data'] = []
    for row in csv_file:
           json_dict['data'].append(row)

    json_dict['columns'] = []
    for colname in colnames:
        column_dict = {}
        column_dict['data'] = colname
        column_dict['name'] = colname
        json_dict['columns'].append(column_dict)
        
    return json_dict

def judge_dtype (np_series, MISSING, UNIQUENESS):

    '''
    입력된 Feature의 데이터 타입을 AutoML에서 사용할 수 있도록 분류/판단하여 리턴

    Args
        df_series : DataFrame 내부의 Feature 데이터 타입 판단을 위해 데이터 값 리스트(series) 입력

    Return
        pandas_dtype : Pandas DataFrame을 통해 최초 판별된 데이터 타입 ex) int64, float64, object, ...
        changed_dtype : 특정 조건들에 의해 pandas_dtype을 수정한 값 ex) int64, float64, object, ...  
        selected_type : 유저가 데이터 타입을 구분하기 쉬운 AutoML 데이터 타입 ex) Numeric, Categorical, Time
        possible_type : 해당 데이터에서 선택이 가능한 AutoML 데이터 타입 리스트 ex) ["Numeric", "Categorical"]   

    '''
    pandas_dtype = str(np_series.dtype)
    possible_type = []

    if ('int' in pandas_dtype):
        if UNIQUENESS <= 1 or UNIQUENESS == 100:
            selected_type = 'Categorical'
            possible_type = ['Numeric', 'Categorical']
            changed_dtype = 'object'

        else:
            selected_type = 'Numeric'
            possible_type = ['Numeric', 'Categorical']
            changed_dtype = pandas_dtype

    elif ('float' in pandas_dtype):
            selected_type = 'Numeric'
            possible_type = ['Numeric', 'Categorical']
            changed_dtype = pandas_dtype

    elif ('boolean' in pandas_dtype):
        selected_type = 'Categorical'
        possible_type = ['Categorical']
        changed_dtype = pandas_dtype

    elif ('time' in pandas_dtype):

        selected_type = 'Time'
        possible_type = ['Time']
        changed_dtype = pandas_dtype


    else:  # ('object' in pandas_dtype)

        selected_type = 'Categorical'
        changed_dtype = pandas_dtype

        try:  # 숫자형 변수인 경우 Numeric으로 변환만 할 수 있게 해주고, 기본은 Categorical 유지
            np_series.astype('float64')
            possible_type = ['Numeric', 'Categorical']

        except:
            possible_type = ['Categorical']

        try:  # Time 변수인 경우 Time 변수로 완전 변환
            np_series.astype('datetime64')
            selected_type = 'Time'
            possible_type = ['Time', 'Categorical']
            changed_dtype = 'datetime64'

        except:
            pass

    return [pandas_dtype, changed_dtype, selected_type, possible_type]

class DataTable:
    """ DataTable 클래스는 """

    def __init__(self, refined_data):
        self.refined_data = refined_data

    # 데이터 자체의 행과 열의 개수 반환
    def file_rowcol(self):
        df = self.refined_data
        row_num, col_num = df.shape[0], df.shape[1]
        return row_num, col_num

    def file_colnames(self):
        df = self.refined_data
        return list(df.columns)

    # 컬럼의 타입 모으기 -> 후에 feature engineering에서 사용
    def file_col_type(self):
        df = self.refined_data
        dtypes_ls = []
        for col in df.columns:
            dtypes_ls.append(df[col].dtypes)
        return dtypes_ls

    # 컬럼별 내용
    # 그 외(Categorical, Bool...): unique value 개수 / most frequent(해당 value명) / most occurrence(most frequent value가 나타난 개수) / NULL
    # Numeric: mean / median / range(범위) / std / NULL
    # [zip(),zip(),zip(),..] 이런형식으로 반환
    def file_info(self):
        df = self.refined_data
        final_return_ls = []
        for col in df.columns:
            data_col = df[col]
            return_ls = []
            # numeric인 경우
            if ('float' in str(data_col.dtypes)) | ('int' in str(data_col.dtypes)):
                name_ls = ['Mean','Median','Range','Std','Null']
                col_mean = round(np.mean(data_col),1); return_ls.append(col_mean)
                col_med = round(np.median(data_col),1); return_ls.append(col_med)
                col_range = '{} - {}'.format(np.min(data_col), np.max(data_col)); return_ls.append(col_range)  # str의 형태로 반환
                col_std = round(np.std(data_col),1); return_ls.append(col_std)
                col_null = sum(data_col.isnull()); return_ls.append(col_null)
                return_dict = {'name_ls':name_ls,'return_ls':return_ls}
            # 그 외 (categorical)인 경우
            else:
                name_ls = ['Unique Values', 'Most Frequent', 'Most Occurence', 'Null','']
                unique_num = len(data_col.unique()); return_ls.append(unique_num)
                most_freq = data_col.value_counts().idxmax(); return_ls.append(most_freq)
                freq_num = data_col.value_counts().max(); return_ls.append(freq_num)
                col_null = sum(data_col.isnull()); return_ls.append(col_null)
                return_ls.append('')
                return_dict = {'name_ls': name_ls, 'return_ls': return_ls}

            final_return_ls.append(zip(*return_dict.values()))
        return final_return_ls

        # plotly version
        # hist: type & x값 & text들 리스트형으로 반환 -> y값의 경우 ''으로
        # bar: type & x값 & y값 & text들 리스트형으로 반환
        # 그래프로 나타내지 않을 것: type & x값 반환. x값의 경우 unique한 개수 -> 나머지의 경우 ''으로 할당
    def file_visualization(self):
        df = self.refined_data
        
        type_ls = []
        X_ls = []
        Y_ls = []

        color_num = 0
        for col in df.columns:
            data_col = df[col]
            if ('float' in str(data_col.dtypes)) | ('int' in str(data_col.dtypes)): #histogram
                # plotly.js에서 nbinsx에서 최대 bins의 개수를 지정해줄 것임
                # And, numerical에서 unique의 개수가 한정되어있다는 것은 categorcial로도 볼 수 있으므로 일단은 이렇게 진행
                type_value = 'Numeric'
                # NaN값이 있을 경우와 없을 경우 
                if sum(data_col.isnull()) > 0:
                    x_values = list(data_col[data_col.notnull()])
                else:
                    x_values = list(data_col)
                y_values = ''

                type_ls.append(type_value)
                X_ls.append(x_values)
                Y_ls.append(y_values)
            else:
                # unique의 개수가 일정 이상 넘어가면 unique value의 값만 띄우게
                if len(data_col.unique()) > 20:
                    type_value = 'Etc'
                    x_values = '{}'.format(len(data_col.unique()))
                    y_values = ''

                    type_ls.append(type_value)
                    X_ls.append(x_values)
                    Y_ls.append(y_values)
                else: # bar
                    tmp = pd.DataFrame(data_col.value_counts()).reset_index()
                    type_value = 'Categorical'
                    x_values = list(tmp.index)
                    y_values = list(tmp[col])

                    type_ls.append(type_value)
                    X_ls.append(x_values)
                    Y_ls.append(y_values)
       
        return type_ls, X_ls, Y_ls


# ==================================================================================================================================

## -- EDA_plot -- ##

# 그래프를 그릴지 말지 판단
# Categorical: uniqueness == 100 | missing == 100
# Numeric: missing == 100
# Times: missing == 100

def judge_draw_element(json_info, column):
    column_type = json_info[column]['selected_type']
    if column_type == 'Categorical':
        if (json_info[column]['uniqueness'] == 100) | (json_info[column]['missing'] == 100):
            return False
        else:
            return True
    else: # Numeric / Time
        if json_info[column]['missing'] == 100:
            return False
        else:
            return True

def judge_draw(json_info, db_csv, col1, col2):

    # col1의 경우 반드시 변수들 중 하나이므로 col2만 처리하면 될 듯
    if col2 == '': # var1만 선택했을 경우
        return judge_draw_element(json_info, col1) # False
    else: # col2 != ''
        if judge_draw_element(json_info, col1) & judge_draw_element(json_info, col2) == False:
            return False
        else: # 두 컬럼의 NaN을 모두 고려했을 때
            temp = db_csv[[col1, col2]]
            temp = db_csv.dropna()
            if temp.shape[0] <= 1:
                return False
            else: 
                return True

def categorical_labeling(x, label_ls):
    ''' Top 5 Label 리스트를 확인하고 x가 포함되어 있지 않은 경우 others로 변환 '''
    if x in label_ls:
        return x
    elif str(x) in label_ls:
        return str(x)
    else:
        return 'others'

# numeric & categorical 한 변수의 경우 숫자 순서대로 label나타나게!
# categorical이라고 정해진 변수들에 대해서만 실행시켜 보기!
def categorical_countvalues(json_info,col,temp_series, get='all'):
    count_values = Counter(list(temp_series)).most_common()
    if len(count_values) > 6:
        label_ls = [value[0] for value in count_values[:5]] 
        # final_labels = label_ls + ['others']
        label_num = 6
        if 'Numeric' not in json_info[col]['possible_type']:
            final_labels = label_ls + ['others']
        else:
            label_ls.sort()
            label_ls = [str(label) for label in label_ls]
            final_labels = label_ls + ['others']

    else:
        label_ls = [value[0] for value in count_values] 
        # final_labels = label_ls
        label_num = len(count_values)
        if 'Numeric' not in json_info[col]['possible_type']:
            final_labels = label_ls
        else:
            label_ls.sort()
            label_ls = [str(label) for label in label_ls]
            final_labels = label_ls

    if get == 'all':
        return label_num, final_labels, label_ls 
    elif get == 'labels':
        return final_labels, label_ls

# bar chart plotly.js에 넘겨줄 값들
# 하나의 categorical data의 distribution을 나타내기 위한 값들
def categorical_chart(json_info, col, col_series):
    temp = pd.DataFrame(col_series)
    label_num, final_labels, label_ls = categorical_countvalues(json_info,col,col_series)
    temp['label'] = temp[col].apply(lambda x: categorical_labeling(x,label_ls))

    data_ls = [] # label의 데이터들(list형태)이 들어갈 list
    for label in final_labels: 
        data_ls.append(temp[temp.label == label].shape[0])
    return [final_labels, data_ls]
    
# timeseries 값을 str으로 변환만
# 얘의 경우 연-월-일로 (연-월-일별 count)
def timeline_chart(col_series):
    time_ascend = col_series.sort_values(ascending=True)
    # '\d{4}-\d{2}-\d{2}': Y-m-d => 현재 (0601)
    # '\d{4}-\d{2}-\d{2} \d{2}': Y-m-d H 
    YMD_ls = list(time_ascend.apply(lambda x: re.search('\d{4}-\d{2}-\d{2}',x).group()))
    count_YMD = Counter(YMD_ls)
    X_ls = list(count_YMD.keys())
    Y_ls = list(count_YMD.values())
    X_range = [X_ls[0], X_ls[-1]]
    # Y_range = [Y_ls[0], Y_ls[-1]]
    Y_range = [min(Y_ls), max(Y_ls)]
    return [X_ls, Y_ls, X_range, Y_range]

# X_ls, scatter_ls, regression_ls, coef, intercept도출
# 짝을 지은 값을 주어야 함
def regression_chart(temp, col1, col2):
    X_ls = list(temp[col1])
    scatter_ls = list(temp[col2])

    model = LinearRegression()
    model.fit(temp[col1].values.reshape(-1,1), temp[col2].values.reshape(-1,1))
    regression_ls = model.predict(temp[col1].values.reshape(-1,1)).reshape(1,-1).tolist()[0]

    coef = round(model.coef_[0][0],4)
    intercept = round(model.intercept_[0],4)

    return [X_ls, scatter_ls, regression_ls, coef, intercept]




# boxplot & violin chart
# horizontal일 경우 x에 값을 주고, vertical인 경우 y에 값을 주면 됨
# label(top5 + others)과 해당 label의 데이터들... 
def boxplot_chart(json_info,temp, col1, col2, graph_type='horizontal'):
    graph_type = graph_type
    if graph_type == 'horizontal':
        num_col = col1 ; cate_col = col2
    elif graph_type == 'vertical':
        num_col = col2 ; cate_col = col1
    
    label_num, final_labels, label_ls = categorical_countvalues(json_info,cate_col,temp[cate_col])
    temp['label'] = temp[cate_col].apply(lambda x: categorical_labeling(x,label_ls))
    data_ls = [] # label의 데이터들(list형태)이 들어갈 list
    for label in final_labels: # others가 맨 뒤에 나오면 좋겠어서 이런식으로 코드 진행
        data_ls.append(list(temp[temp.label == label][num_col]))
    return [label_num, final_labels, data_ls]

# stacked histogram chart 
# col1(X): numeric & col2(Y): categorical 
def stacked_histogram_chart(json_info,temp, col1, col2): 
    label_num, final_labels, label_ls = categorical_countvalues(json_info,col2,temp[col2])
    temp['label'] = temp[col2].apply(lambda x: categorical_labeling(x,label_ls))
    data_ls = [] # label의 데이터들(list형태)이 들어갈 list
    for label in final_labels: 
        data_ls.append(list(temp[temp.label == label][col1]))
    return [label_num, final_labels, data_ls]

# contingency table (crosstab)
# percent (소수점 두자리수까지)
def heatmap_chart(json_info,temp, col1, col2):
    final_labels1, label_ls1 = categorical_countvalues(json_info,col1,temp[col1], get='labels')
    final_labels2, label_ls2 = categorical_countvalues(json_info,col2,temp[col2], get='labels')

    temp['label1'] = temp[col1].apply(lambda x: categorical_labeling(x,label_ls1))
    temp['label2'] = temp[col2].apply(lambda x: categorical_labeling(x,label_ls2))

    crosstab = pd.crosstab(temp['label2'], temp['label1'], margins=False)
    crosstab_percent = np.round((crosstab/temp.shape[0])*100,2)

    X_ls = list(crosstab.columns)
    Y_ls =  final_labels2 ; Y_ls.reverse()
    data_ls = []
    for y_category in Y_ls:
        data_ls.append(crosstab_percent.loc[y_category,:].values.tolist())    

    return [X_ls, Y_ls, data_ls]

# side bar / stacked bar
# count
# X_cate_num * Y_cate_num > 20: side bar / else: stacked bar
# mode값 (group / stack) return 필요
def side_stacked_chart(json_info,temp, col1, col2):   
    label_num1, final_labels1, label_ls1 = categorical_countvalues(json_info,col1,temp[col1])
    label_num2, final_labels2, label_ls2 = categorical_countvalues(json_info,col2,temp[col2])

    if label_num1 * label_num2 < 20:
        mode = 'group'
    else: # bar가 20개 이상일 경우 stacked bar로!
        mode = 'stack'

    temp['label1'] = temp[col1].apply(lambda x: categorical_labeling(x,label_ls1))
    temp['label2'] = temp[col2].apply(lambda x: categorical_labeling(x,label_ls2))

    crosstab = pd.crosstab(temp['label1'], temp['label2'], margins=False)

    X_ls = list(crosstab.index)
    Y_ls = final_labels2
    data_ls = []
    for y_category in Y_ls:
        data_ls.append(crosstab[y_category].values.tolist())  

    return [X_ls, Y_ls, label_num2, data_ls, mode]

# overlapping chart & stacking line chart
# col1: Time / col2: Categorical
# Time: 연-월-일로 change
def TC_line_chart(json_info,temp, col1, col2):
    temp = temp.sort_values(by=col1, ascending=True)
    temp['label1'] = temp[col1].apply(lambda x: re.search('\d{4}-\d{2}-\d{2}',x).group()) # time

    label_num, final_labels, label_ls = categorical_countvalues(json_info,col2,temp[col2])
    temp['label2'] = temp[col2].apply(lambda x: categorical_labeling(x,label_ls)) # category

    crosstab = pd.crosstab(temp['label2'], temp['label1'], margins=False)

    X_ls = list(crosstab.columns) # time
    Y_ls = final_labels # y_category
    data_ls = []
    for y_category in Y_ls:
        data_ls.append(crosstab.loc[y_category,:].values.tolist()) 
    X_range = [X_ls[0],X_ls[-1]]

    return [X_ls, Y_ls, data_ls, label_num, X_range]

# Categorical & Numerical (Ver.3) 일 때
def distplot(json_info,temp,col1,col2):
    label_num, final_labels, label_ls = categorical_countvalues(json_info,col2,temp[col2])
    temp['label'] = temp[col2].apply(lambda x: categorical_labeling(x,label_ls))
    data_ls = [] # label의 데이터들(list형태)이 들어갈 list

    i = 0
    for label in final_labels: # others가 맨 뒤에 나오면 좋겠어서 이런식으로 코드 진행
        temp2 = temp[temp.label == label].reset_index(drop=True)
        ax = sns.distplot(temp2[col1],hist=False)
        line = ax.lines[i].get_data()
        data_x, data_y = line[0].tolist(), line[1].tolist()
        data_ls.append([data_x,data_y])
        i += 1
    plt.cla()
    return [label_num, final_labels, data_ls]


# 그래프 보조 text 
def graph_descrip(type, json_info, col1, col2='', params=''):
    if col2 == '':

        TITLE = ''
        if type == 'histogram':
            TEXT = '위 그래프는 수치형 변수 <b>['+ col1 + ']</b>의 데이터 분포를 나타내는 히스토그램(Histogram) 입니다.<br>'\
                     + '히스토그램은 수치형 데이터의 범위를 일정 간격으로 나누고 빈도를 그래프의 높이로 나타냅니다. '\
                     + '그래프를 통해 구간별 데이터 빈도를 비교하거나 전체 데이터 분포를 확인해보세요.<br>'\
                     + '<b>[' + col1 + ']</b>의 전체 데이터 대비 변수 데이터 결측률은 <b>'+ str(json_info[col1]['missing']) +'%</b> 입니다. '\
                     + '결측된 데이터가 많을수록 해당 변수에 대한 데이터 신뢰성이 떨어지니 주의해야합니다.'
        elif type == 'bar':
            TEXT = '위 그래프는 범주형 변수 <b>['+ col1 + ']</b>의 데이터 분포를 나타내는 막대 그래프(Bar Chart) 입니다.<br>'\
                     + '막대 그래프는 변수의 범주 값들을 나열하고 데이터 빈도를 막대의 높이로 나타냅니다. '\
                     + '그래프를 통해 범주별 데이터의 빈도를 비교하거나, 전체적인 데이터의 분포를 확인해보세요.<br>'\
                     + '<b>[' + col1 + ']</b>의 전체 데이터 대비 변수 데이터 결측률은 <b>'+ str(json_info[col1]['missing']) +'%</b> 입니다. '\
                     + '결측된 데이터가 많을수록 해당 변수에 대한 데이터 신뢰성이 떨어지니 주의해야합니다.'
        elif type == 'time':
            TEXT = '위 그래프는 시간형 변수 <b>['+ col1 + ']</b>의 데이터 분포를 나타내는 꺾은선 그래프(Line Chart) 입니다.<br>'\
                     + '꺾은선 그래프는 시간의 변화를 기준으로 데이터 빈도를 선 높이로 나타냅니다. '\
                     + '그래프를 통해 시간 흐름에 따른 데이터 변화를 비교하거나 전체적인 데이터 분포를 확인해보세요.<br>'\
                     + '<b>[' + col1 + ']</b>의 전체 데이터 대비 변수 데이터 결측률은 <b>'+ str(json_info[col1]['missing']) +'%</b> 입니다. '\
                     + '결측된 데이터가 많을수록 해당 변수에 대한 데이터 신뢰성이 떨어지니 주의해야합니다.'

    else:
        # 1) scatter + regression
        # params = [coef , intercept]
        if type == 'regression':
            TEXT = '위 그래프는 <b>['+ col1 + ']</b>와(과) <b>[' + col2 + ']</b>의 데이터 분포를 종합적으로 나타내는 산점도(Scatter Plot) 및 선형 회귀 그래프(Linear Regression) 입니다. ' \
                 + '산점도는 특정 데이터에 대한 두 변수 간의 관계를 점의 위치로 나타내는 그래프로 두 변수의 데이터 분포와 관계를 한눈에 보여줍니다.  '\
                 + '선형 회귀 그래프는 독립변수 ['+col1+']과 종속변수 ['+col2+']의 상관관계를 분석하여 직선으로 나타낸 그래프로, 현재 두 변수의 관계식은 '\
                 + '<b>{} = {} * {} + {}</b> 입니다. '.format(col2, round(params[0], 2), col1, round(params[1], 1))\
                 + '해당 관계식은 {}은(는) {}와 {} 비율로 비례하며 편향 값인 {}만큼 더해 유추할 수 있다는 것을 의미합니다. '.format(col2, col1, params[0], params[1])
            TITLE = 'Scatter & Regression Plot of {} vs {}'.format(col1,col2)
        
        # 2) 2d-density
        elif type == '2d-density':
            TEXT = '위 그래프는 <b>['+ col1 + ']</b>와(과) <b>[' + col2 + ']</b>의 데이터 밀도를 종합적으로 나타내는 이차원 밀도 그래프(2D Density Plot)입니다. ' \
                 + '이차원 밀도 그래프는 두 변수의 값에 따라 데이터가 밀집되어 있는 정도를 나타내는 밀도 추정치를 산출하고, 같은 추정치를 선으로 연결하여 등고선 같이 나타냅니다. '\
                 + '또한 구분된 영역의 데이터 밀도를 색으로 구분하여, 점 분포와 영역의 색을 통해 데이터 밀집 분포를 한눈에 확인 할 수 있도록 도와줍니다. '\
                 + '추가적으로 각 변수의 분포를 나타내는 히스토그램과 밀도 그래프를 서로 비교하여 데이터 밀도를 좀더 세밀하게 확인해볼 수 있습니다. '
            TITLE = '2D Density Plot of {} vs {}'.format(col1,col2)

        # 3-1) vertical boxplot
        elif type == 'v-boxplot':
            TEXT = '위 그래프는 수치형 변수 <b>[{}]</b>의 분포를 범주형 변수 <b>[{}]</b> 값으로 구분하여 나타낸 상자 그래프(Box Plot)입니다. '.format(col2, col1)\
                 + '상자 그래프는 데이터 분포를 사분위수 기준으로 보여주는 간단하지만 매우 효과적인 그래프입니다. '\
                 + '그래프 내부 상자는 하위 25%값 <b>q1</b>, 50%값 <b>q2</b>, 75%값 <b>q3</b>를 기준으로 중간 데이터 범위를 나타내며 위아래 울타리는 통계적 데이터 평균치 범위를 보여줍니다. '\
                 + '또한 통계적으로 평균치를 벗어난 <b>이상치(Outlier)</b>를 울타리 밖 점으로 나타내므로 이를 쉽게 확인할 수 있습니다. '
            TITLE = 'Box Plot of {} Grouped by {}'.format(col2, col1)

        # # 3-2) horizontal boxplot
        # elif type == 'h-boxplot':
        #     TEXT = '해당 그래프는 <b>[{}]의 항목별로 [{}]의 분포</b>를 나타낸 박스플롯(Box Plot)입니다. '.format(col2, col1)\
        #          + '박스플롯은 데이터의 분포를 하나의 박스로 나타내는 그래프로, 간단하지만 데이터의 수치적인 분석과 이상치를 한번에 보여주여 매우 효과적인 그래프입니다. '\
        #          + '점으로 표시된 데이터의 경우 이상치라는 의미이므로 주의깊게 보시길 바랍니다. '
        #     TITLE = 'Box Plot of {} Grouped by {}'.format(col1, col2)  

        # 4) vertical violin
        elif type == 'violin':
            TEXT = '위 그래프는 수치형 변수 <b>[{}]</b>의 분포를 범주형 변수 <b>[{}]</b> 값으로 구분하여 나타낸 바이올린 그래프(Violin Plot)입니다. '.format(col2, col1)\
                 + '바이올린 그래프는 상자 그래프 같이 최대값 Max, 최소값 Min, 하위 25%값 <b>q1</b>, 50%값 <b>q2</b>, 75%값 <b>q3</b>을 나타내어 주요 데이터 값들을 확인할 수 있을 뿐만 아니라, 데이터의 빈도를 나타내는 KDE 값을 넓이로 나타내어 범주별 데이터의 분포를 한눈에 비교할 수 있습니다. '\
                 + '또한 통계적으로 평균치를 벗어난 <b>이상치(Outlier)</b>를 점으로 나타내므로 이를 쉽게 확인할 수 있습니다.'
            TITLE = 'Violin Plot of {} Grouped by {}'.format(col2, col1)

        # 5) contigency table
        elif type == 'heatmap':
            TEXT = '위 그래프는 범주형 변수 <b>[{}]</b>와(과) <b>[{}]</b>의 관계를 나타낸 히트맵 그래프(Heatmap Chart)입니다. '.format(col1, col2)\
                 + '히트맵 그래프는 두 개의 범주형 변수에 따른 경우의 수를 표 형식으로 나타내며 데이터의 비중을 색상으로 변환시켜 시각적으로 표현하는 그래프입니다. '\
                 + '데이터의 비중이 클수록 진한 색을, 작을수록 옅은 색을 띠고 있어 이를 한눈에 비교 할 수 있습니다. 히트맵 그래프 색깔을 통해 데이터 비중의 변화를 확인해보세요. '
            TITLE = 'Heatmap Chart of {} vs {}'.format(col1, col2)

        # 6) stacked histogram
        elif type == 'stacked-histogram':
            TEXT = '위 그래프는 수치형 변수 <b>[{}]</b>와(과) 범주형 변수 <b>[{}]</b>의 데이터 분포를 나타내는 겹친 히스토그램(Stacked Histogram)입니다. '.format(col1, col2)\
                 + '겹친 히스토그램은 일반 히스토그램과 같이 <b>[{}]</b>의 구간별 빈도수로 구성된 그래프지만, 그 값을 <b>[{}]</b>의 범주별로 나눠서 겹쳐진 형태로 나타냅니다. '.format(col1, col2)\
                 + 'Stacked라는 단어와 같이 위로 겹쳐진 형태이기 때문에 전체적인 [{}]의 분포 뿐만 아니라, [{}]의 범주별 [{}]의 분포도 확인 할 수 있습니다. '.format(col1, col2, col1)\
                 + '히스토그램 구간별 [{}] 범주값 비율 변화를 주의깊게 확인해야 합니다.'.format(col2)
            TITLE = 'Stacked Histogram of {} Grouped by {}'.format(col1, col2)

        # 7) side bar / stacked bar
        elif type == 'side-bar':
            TEXT = '위 그래프는 범주형 변수 <b>[{}]</b>의 분포를 <b>[{}]</b> 값으로 구분하여 나타낸 그룹 막대 그래프(Bar Chart)입니다. '.format(col1, col2)\
                 + '그룹 막대 그래프는 일반 막대그래프와 같이 [{}]의 범주별 데이터 빈도를 나타내는 그래프지만, 그 값을 [{}]의 범주에 따라 다른 색으로 구분지어 나타냅니다. '.format(col1, col2)\
                 + '전체적인 [{}]의 데이터 빈도 뿐만 아니라 [{}]의 범주별 [{}]의 분포도 한 눈에 확인 할 수 있습니다. '.format(col1, col2, col1)\
                 + '[{}]값에 따른 [{}] 데이터 빈도/비중 변화를 주의깊게 확인해야 합니다.'.format(col1, col2)
            TITLE = 'Bar Chart of {} Grouped by {}'.format(col1, col2)

        # 8) time line chart
        elif type == 'time-line-chart':
            TEXT = '위 그래프는 시간형 변수 <b>[{}]</b>와 수치형 변수 <b>[{}]</b>의 관계를 꺾은선 그래프(Line Chart)로 나타낸 것입니다. '.format(col1, col2)\
                 + '꺾은선 그래프는 시간형 변수가 나타내는 특정 시각을 기준으로 [{}] 값을 그래프의 높낮이로 나타냅니다. '.format(col2)\
                 + '그래프를 통해 시간형 변수 [{}]의 변화에 따라 [{}] 값의 연속적인 변화 상태와 방향을 확인해보고, 시간 변화에 따른 [{}] 값의 증감을 예측해보세요.'.format(col1, col2, col2)
            TITLE = 'Time Line Chart of {} with {}'.format(col2, col1)

        # 9) overlapping chart (time)
        elif type == 'time-area-chart':
            TEXT = '위 그래프는 시간형 변수 <b>[{}]</b>와 범주형 변수 <b>[{}]</b>의 관계를 면적 그래프(Area Chart)로 나타낸 것입니다. '.format(col1, col2)\
                 + '면적 그래프는 시간형 변수가 나타내는 특정 시각을 기준으로 [{}] 범주값의 빈도를 그래프의 높낮이로 나타냅니다. '.format(col2)\
                 + '면적 그래프를 통해 시간 [{}]의 변화에 따라 [{}] 범주별 데이터 빈도의 변화 상태와 방향을 확인해보고, 이를 활용해서 시간 변화에 따른 빈도 증감를 예측해보세요.'.format(col1, col2)
            TITLE = 'Time Area Chart of {} with {}'.format(col2, col1)

        # 10) stacked bar chart (time)
        elif type == 'time-bar-chart':
            TEXT = '위 그래프는 시간형 변수 <b>[{}]</b>와 범주형 변수 <b>[{}]</b>의 관계를 겹친 막대그래프(Stacked Bar Chart)로 나타낸 것입니다. '.format(col1, col2)\
                 + '해당 그래프는 시간형 변수가 나타내는 특정 시각을 기준으로 [{}] 범주값의 빈도를 막대 길이로 나타냅니다. '.format(col2)\
                 + '겹친 막대그래프를 통해 시간 [{}]의 변화에 따라 [{}] 범주별 데이터 빈도와 비중의 변화 상태를 확인해보고, 이를 활용해서 시간 변화에 따른 빈도 증감을 예측해보세요. '.format(col1, col2)
            TITLE = 'Time Stacked Bar Chart of {} with {}'.format(col2, col1)

        # 11) distplot
        elif type == 'distplot':
            TEXT = '위 그래프는 수치형 변수 <b>[{}]</b>의 데이터 밀도를 범주형 변수 <b>[{}]</b> 값으로 구분하여 나타낸 커널밀도 곡선(Kernel Density Curve)입니다. '.format(col1, col2)\
                 + '커널밀도 곡선은 히스토그램을 변환하여 데이터의 연속적인 확률분포(Probability Distribution)를 나타내는 그래프로 전체 면적의 합은 1입니다. '\
                 + '그래프를 통해 데이터의 편향/편차를 비롯한 변수 값에 따른 확률분포의 변화를 확인해보세요. '\
                 + '특히 <b>[{}]</b> 범주별 <b>[{}]</b> 확률분포가 차이를 보이는 경우에는 두 변수의 관계를 주의깊게 확인해야 합니다. '.format(col2, col1)
            TITLE = 'Kernel Density Curve of {} Grouped by {}'.format(col1, col2)

    return TITLE, TEXT