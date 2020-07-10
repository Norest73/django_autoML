from django.shortcuts import render, redirect
from django.urls import reverse
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.contrib import messages

import os, time
from ipware.ip import get_client_ip
from .about_data import *
from .Report import *

def index(request):
    ''' Main Page '''
    return render(request, './MachineLearning/index.html')


def DataTable_view(request):
    ''' Make the Date Table and Visualize the simple plots '''
    client_ip = get_client_ip(request)[0]

    if request.method == 'POST':  # file upload method == 'POST'

        start_time = time.time()
        uploaded_file = request.FILES['file_CSV']  # <html> input id='file' (type='file')
        fs = FileSystemStorage()                   # default path = /media/
        file_name = fs.save(client_ip, uploaded_file)  # 업로드 파일 IP를 이름으로 저장
        file_path = os.path.abspath(os.getcwd() + fs.url(file_name))

        print('BackEnd 처리 진행중 (1/5)')

        # media CSV 파일 DataFrame 형태로 로딩 후 빈값 nan 처리
        refined = pd.read_csv(file_path, low_memory=False)
        refined.replace('', np.nan, inplace=True)

        print('BackEnd 처리 진행중 (2/5)')

        # META INFO DATA JSON 파일로 저장 (selected_type, ...)
        refined_info = {}

        for col in refined.columns:
            
            TOTAL_NUM = len(refined[col])
            NULL_NUM = refined[col].isnull().sum()
            UNIQUE_NUM = len(refined[col].unique())

            if TOTAL_NUM == NULL_NUM :
                refined.drop(col, axis=1, inplace=True)

            else:
                # 결측치 및 UNIQUE 상수 산출
                MISSING = round(100*(NULL_NUM / TOTAL_NUM), 2)
                UNIQUENESS = round(100*(UNIQUE_NUM / (TOTAL_NUM-NULL_NUM)), 2)
                
                refined_info[col] = {}
                dtype_group = judge_dtype(refined[col].to_numpy(), MISSING, UNIQUENESS)
                
                refined_info[col]['pandas_dtype'] = dtype_group[0]
                refined_info[col]['changed_dtype'] = dtype_group[1]
                refined_info[col]['selected_type'] = dtype_group[2]
                refined_info[col]['possible_type'] = dtype_group[3]
                refined_info[col]['uniqueness'] = UNIQUENESS
                refined_info[col]['missing'] = MISSING

                # Time인 경우 Plot을 그려주기 위해 데이터 타입 변경
                if dtype_group[2] == 'Time':
                    refined[col] = refined[col].astype('str')

        print('BackEnd 처리 진행중 (3/5)')

        # 정제된 데이터을 DB 폴더에 저장, 이후부터 해당 파일을 통해 데이터 업데이트 진행
        db_path = os.path.abspath(os.path.join(settings.DB_ROOT, '{}.csv'.format(client_ip)))
        refined.to_csv(db_path, index=False) 

        print('BackEnd 처리 진행중 (4/5)')

        with open(os.path.abspath(os.path.join(settings.DB_ROOT, '{}_meta-info.json'.format(client_ip))),'w') as f:
            json.dump(refined_info, f)

        print('BackEnd 처리 완료 (5/5)')

        if time.time() - start_time >= 1:
            return TargetSelection(request)

        else:
            # Data Table Upload를 위해 DB CSV 파일을 JSON 파일로 변환 후 저장
            json_dict = csv_to_json(db_path)
            with open(os.path.abspath(os.path.join(settings.DB_ROOT, '{}.json'.format(client_ip))),'w') as data_json:
                json.dump(json_dict, data_json)

            DT = DataTable(refined)
            colnames = DT.file_colnames()
            type_ls, X_ls, Y_ls = DT.file_visualization()
            file_infos = DT.file_info()
            dtypes_ls = DT.file_col_type()
            rows, col_nums = DT.file_rowcol()

            return render(request, './MachineLearning/DataTable.html', {'colnames': colnames,'file_infos':file_infos,
                                                                        'dtypes_ls':dtypes_ls,'col_nums': col_nums,
                                                                        'type_ls':type_ls, 'X_ls':X_ls, 'Y_ls':Y_ls, 'IP': client_ip})

def TargetSelection(request) :
    ''' Get the Target Column & Confirm the columns dtype info '''

    client_ip = get_client_ip(request)[0]
    db_path = os.path.abspath(os.path.join(settings.DB_ROOT, '{}_meta-info.json'.format(client_ip)))
    df_info = pd.read_json(db_path)
    selected_types = []
    possible_types = []
    description = []

    for col in list(df_info.columns):
        selected_types.append(df_info[col]['selected_type'])
        possible_types.append(df_info[col]['possible_type'])

        if df_info[col]['possible_type'] == ['Numeric', 'Categorical']:
            description.append("실수 데이터로 구성된 숫자형 변수입니다. 그러나 데이터가 단순 숫자값이 아닌 특정 범주를 나타내는 경우에는 범주형 변수로 선택해야 합니다.")

        elif df_info[col]['possible_type'] == ['Categorical']:
            description.append("데이터의 정해진 특징이나 구분에 대한 값으로 구성된 범주형 변수입니다. 범주 값에 따른 데이터 변화를 확인 할 수 있습니다.")

        elif 'Time' in df_info[col]['possible_type']:
            description.append("특정 날짜나 시간을 나타내는 시간형 변수입니다. 다른 변수와 함께 분석하여 시간 흐름에 따른 변화를 확인 할 수 있습니다.")

    return render(request, './MachineLearning/TargetSelection.html', {'colnames': list(df_info.columns), 'selected_types': selected_types,
                                                                        'possible_types': possible_types, 'description': description})


def Exploratory(request) :

    client_ip = get_client_ip(request)[0]
    db_path = os.path.abspath(os.path.join(settings.DB_ROOT, '{}_meta-info.json'.format(client_ip)))
    df_info = pd.read_json(db_path)
    df_columns = list(df_info.columns)

    # Loading the Target Feature, Types selected by User
    target = request.POST.get('target-feature')

    # Target Feature를 체크하는 'Target': True/False 추가 & 데이터 타입 선택에 따라 selected_type, changed_dtype 값 변경
    target_feature = []
    for i, col in enumerate(df_columns) :

        # selected_type이 유저에 의해 변경되었을 경우, 그것에 맞도록 changed_dtype 변경
        user_type = request.POST.get('dtype' + str(i))
        if df_info[col]['selected_type'] != user_type :
            df_info[col]['selected_type'] = user_type

            if df_info[col]['selected_type'] == 'Numeric':
                df_info[col]['changed_dtype'] = 'int64' if 'int' in df_info[col]['pandas_dtype'] else 'float64'

            elif df_info[col]['selected_type'] == 'Time':
                df_info[col]['changed_dtype'] = 'datetime64'

            else : # df_info[col]['selected_dtype'] == 'Categorical', 'TIme':
                df_info[col]['changed_dtype'] = 'object'

        # Target Feature에 대한 정보를 나타내는 리스트 구성 ex. [False, True, False, False, ...]
        tf = True if col == target else False
        target_feature.append(tf)
        
    df_info.loc['target'] = target_feature
    df_info.to_json(os.path.join(settings.DB_ROOT, '{}_meta-info.json'.format(client_ip)))

    # Selectbox 구성을 위한 Feature list 회신, Var2에는 Time변수는 선택할 수 없음
    Var1_list = df_columns
    Var2_list = [col for col in df_columns if df_info[col]['selected_type'] != 'Time']

    return render(request, './MachineLearning/EDA_main.html', {'Var1':Var1_list,'Var2':Var2_list})

def ExploratoryPlot(request) :

    client_ip = get_client_ip(request)[0]

    # 변수를 선택했는지 여부에 따라 처음에 빈 화면, 변수를 선택 했을 때는 EDA 결과를 리턴
    if request.method == 'POST':
        col1 = request.POST.get('var1')
        col2 = request.POST.get('var2')

        # DB에서 csv 데이터/info 데이터 로딩
        df_refined = pd.read_csv(os.path.abspath(os.path.join(settings.DB_ROOT, '{}.csv'.format(client_ip))), low_memory=False)
        json_path = os.path.abspath(os.path.join(settings.DB_ROOT, '{}_meta-info.json'.format(client_ip)))
        with open(json_path, 'r') as f:
            json_info = json.load(f)

        # -- 2-1) 공통 //if & else 문// 2-2) only Individual 2-3) Group -- #
        # -- 공통 - Both / Individual - Ind / Group - Gr 변수에 해당 내용 모두 넣기

        # 유저가 선택한 var1, var2의 그래프를 그릴 수 있는지 데이터 조건 확인 (Unique, Missing, 外 (Same도 추가 해야되나?))
        if judge_draw(json_info, df_refined, col1, col2) == False:

            Version = 'none_plot'  # Version 변수를 통해 어떠한 그래프를 그릴지 제어
            return render(request, './MachineLearning/EDA_plot.html', {'Version':Version, 'col1':'None', 'col2':'None', 
                        'X_DIST': 'None', 'Y_DIST': 'None', 'XY_DIST1': 'None', 'XY_DIST2': 'None'})


        # 유저가 선택한 var1, var2의 그래프를 그릴 있는 경우
        else:  
            # 1) 공통 (Var1 distribution graph)
            col1_type = json_info[col1]['selected_type']
            col1_data = df_refined[col1].dropna()

            # 데이터 타입에 따라 Numeric, Categorical, Time 구분하여 그래프 그림
            if col1_type == 'Numeric': # histogram
                Version = 'only_Numeric'
                X_DIST = list(col1_data)
                _, X_TEXT = graph_descrip('histogram', json_info, col1)

            elif col1_type == 'Categorical': # bar chart
                Version = 'only_Categorical'
                X_DIST = categorical_chart(json_info, col1, col1_data)
                _, X_TEXT = graph_descrip('bar', json_info, col1)

            elif col1_type == 'Time': # timeseries
                Version = 'only_Time'
                X_DIST = timeline_chart(col1_data)
                _, X_TEXT = graph_descrip('time', json_info, col1)

            # 2) only Individual 
            if (col2 == '') | (col1 == col2): # Var2값이 None 
                return render(request, './MachineLearning/EDA_plot.html', {'Version': Version, 'col1':col1, 'col2':col2, 'X_DIST': X_DIST, 'Y_DIST': 'None',
                                                        'XY_DIST1': 'None', 'XY_DIST2': 'None',
                                                        'X_TEXT': X_TEXT, 'Y_TEXT':'None', 'XY_TEXT1':'None', 'XY_TEXT2':'None',
                                                        'XY_TITLE1': 'None', 'XY_TITLE2': 'None'})

            # 3) Group (Var2 distribution graph + 6 versions)
            else: 
                col2_type = json_info[col2]['selected_type']
                col2_data = df_refined[col2].dropna()

                if (col1_type == 'Numeric') & (col2_type == 'Numeric'): # version 1
                    Version = 'Numeric_Numeric'
                    temp = df_refined[[col1,col2]]
                    temp = temp.dropna()
                    temp.reset_index(drop=True, inplace=True)

                    # Y-histogram
                    Y_DIST = list(col2_data)
                    # scatter + regression
                    XY_DIST1 = regression_chart(temp, col1, col2)
                    # 2d density
                    XY_DIST2 = [list(temp[col1]), list(temp[col2])] # X_ls, Y_ls


                    # TEXT
                    _, Y_TEXT = graph_descrip('histogram', json_info, col2)
                    XY_TITLE1, XY_TEXT1 = graph_descrip('regression', json_info, col1, col2=col2, params=[XY_DIST1[3],XY_DIST1[4]])
                    XY_TITLE2, XY_TEXT2 = graph_descrip('2d-density', json_info, col1, col2=col2)

                    return render(request, './MachineLearning/EDA_plot.html', {'Version': Version, 'col1':col1, 'col2': col2,'X_DIST': X_DIST, 'Y_DIST': Y_DIST,
                                                             'XY_DIST1': XY_DIST1, 'XY_DIST2': XY_DIST2,
                                                             'X_TEXT': X_TEXT, 'Y_TEXT':Y_TEXT, 'XY_TEXT1':XY_TEXT1, 'XY_TEXT2':XY_TEXT2,
                                                             'XY_TITLE1': XY_TITLE1, 'XY_TITLE2': XY_TITLE2})


                elif (col1_type == 'Numeric') & (col2_type == 'Categorical'):
                    Version = 'Numeric_Categorical'
                    temp = df_refined[[col1,col2]]
                    temp = temp.dropna()
                    temp.reset_index(drop=True, inplace=True)

                    # Y-pie/donut/bar chart
                    Y_DIST = categorical_chart(json_info, col2, col2_data)
                    # horizontal boxplot & violin
                    XY_DIST1 = distplot(json_info, temp, col1, col2)
                    # stacked histogram
                    XY_DIST2 = stacked_histogram_chart(json_info, temp, col1, col2)

                    # TEXT
                    _, Y_TEXT = graph_descrip('bar', json_info, col2)
                    XY_TITLE1, XY_TEXT1 = graph_descrip('distplot', json_info, col1, col2=col2)
                    XY_TITLE2, XY_TEXT2 = graph_descrip('stacked-histogram', json_info, col1, col2=col2)

                    return render(request, './MachineLearning/EDA_plot.html', {'Version': Version, 'col1':col1, 'col2': col2,'X_DIST': X_DIST, 'Y_DIST': Y_DIST,
                                                             'XY_DIST1': XY_DIST1, 'XY_DIST2': XY_DIST2,
                                                             'X_TEXT': X_TEXT, 'Y_TEXT':Y_TEXT, 'XY_TEXT1':XY_TEXT1, 'XY_TEXT2':XY_TEXT2,
                                                             'XY_TITLE1': XY_TITLE1, 'XY_TITLE2': XY_TITLE2})

                elif (col1_type == 'Categorical') & (col2_type == 'Numeric'):
                    Version = 'Categorical_Numeric'
                    temp = df_refined[[col1,col2]]
                    temp = temp.dropna()
                    temp.reset_index(drop=True, inplace=True)

                    # Y-histogram
                    Y_DIST = list(col2_data)
                    # boxplot vertical & violin
                    XY_DIST1 = boxplot_chart(json_info, temp, col1, col2, graph_type='vertical')

                    # TEXT
                    _, Y_TEXT = graph_descrip('histogram', json_info, col2)
                    XY_TITLE1, XY_TEXT1 = graph_descrip('v-boxplot', json_info, col1, col2=col2)
                    XY_TITLE2, XY_TEXT2 = graph_descrip('violin', json_info, col1, col2=col2)

                    return render(request, './MachineLearning/EDA_plot.html', {'Version':Version, 'col1':col1, 'col2': col2, 'X_DIST': X_DIST, 'Y_DIST':Y_DIST,
                                                            'XY_DIST1': XY_DIST1, 'XY_DIST2': XY_DIST1,
                                                            'X_TEXT': X_TEXT, 'Y_TEXT':Y_TEXT, 'XY_TEXT1':XY_TEXT1, 'XY_TEXT2':XY_TEXT2,
                                                            'XY_TITLE1': XY_TITLE1, 'XY_TITLE2': XY_TITLE2})

                elif (col1_type == 'Categorical') & (col2_type == 'Categorical'):
                    Version = 'Categorical_Categorical'
                    temp = df_refined[[col1,col2]]
                    temp = temp.dropna()
                    temp.reset_index(drop=True, inplace=True)
                    # Y-pie/donut/bar chart
                    Y_DIST = categorical_chart(json_info, col2, col2_data)
                    # contingency table (crosstab) - heatmap
                    XY_DIST1 = heatmap_chart(json_info, temp, col1, col2)
                    # side bar / stacked bar
                    XY_DIST2 = side_stacked_chart(json_info, temp, col1, col2)

                    # TEXT
                    _, Y_TEXT = graph_descrip('bar', json_info, col2)
                    XY_TITLE1, XY_TEXT1 = graph_descrip('heatmap', json_info, col1, col2=col2)
                    XY_TITLE2, XY_TEXT2 = graph_descrip('side-bar', json_info, col1, col2=col2)

                    return render(request, './MachineLearning/EDA_plot.html', {'Version':Version, 'col1':col1, 'col2': col2, 'X_DIST': X_DIST, 'Y_DIST':Y_DIST, 
                                                             'XY_DIST1': XY_DIST1, 'XY_DIST2': XY_DIST2,
                                                             'X_TEXT': X_TEXT, 'Y_TEXT':Y_TEXT, 'XY_TEXT1':XY_TEXT1, 'XY_TEXT2':XY_TEXT2,
                                                             'XY_TITLE1': XY_TITLE1, 'XY_TITLE2': XY_TITLE2})

                elif (col1_type == 'Time') & (col2_type == 'Numeric'):
                    Version = 'Time_Numeric'
                    temp = df_refined[[col1,col2]]
                    temp = temp.dropna()
                    temp.reset_index(drop=True, inplace=True)
                    # Y-histogram
                    Y_DIST = list(col2_data)
                    # time line chart
                    X_ls = list(temp[col1])
                    Y_ls = list(temp[col2])
                    XY_DIST = [X_ls, Y_ls, [X_ls[0], X_ls[-1]], [Y_ls[0], Y_ls[-1]]]

                    # TEXT
                    _, Y_TEXT = graph_descrip('histogram', json_info, col2)
                    XY_TITLE1, XY_TEXT1 = graph_descrip('time-line-chart', json_info, col1, col2=col2)

                    return render(request, './MachineLearning/EDA_plot.html', {'Version':Version, 'col1':col1, 'col2': col2, 'X_DIST': X_DIST, 'Y_DIST': Y_DIST,
                                                             'XY_DIST1': XY_DIST, 'XY_DIST2': 'None',
                                                             'X_TEXT': X_TEXT, 'Y_TEXT':Y_TEXT, 'XY_TEXT1':XY_TEXT1, 'XY_TEXT2':'None',
                                                             'XY_TITLE1': XY_TITLE1, 'XY_TITLE2': 'None'})

                elif (col1_type == 'Time') & (col2_type == 'Categorical'):
                    Version = 'Time_Categorical'
                    temp = df_refined[[col1,col2]]
                    temp = temp.dropna()
                    temp.reset_index(drop=True, inplace=True)
                    # Y-pie/donut/bar chart
                    Y_DIST = categorical_chart(json_info, col2, col2_data)
                    # overlapping chart & stacked bar OR stacked line chart
                    XY_DIST = TC_line_chart(json_info, temp, col1, col2)

                    # TEXT
                    _, Y_TEXT = graph_descrip('bar', json_info, col2)
                    XY_TITLE1, XY_TEXT1 = graph_descrip('time-area-chart', json_info, col1, col2=col2)
                    XY_TITLE2, XY_TEXT2 = graph_descrip('time-bar-chart', json_info, col1, col2=col2)

                    return render(request, './MachineLearning/EDA_plot.html', {'Version':Version, 'col1':col1, 'col2': col2, 'X_DIST': X_DIST, 'Y_DIST':Y_DIST, 
                                                             'XY_DIST1': XY_DIST, 'XY_DIST2': XY_DIST,
                                                             'X_TEXT': X_TEXT, 'Y_TEXT':Y_TEXT, 'XY_TEXT1':XY_TEXT1, 'XY_TEXT2':XY_TEXT2,
                                                             'XY_TITLE1': XY_TITLE1, 'XY_TITLE2': XY_TITLE2})

    # 초기 화면 (빈 화면)
    else:
        return render(request, './MachineLearning/EDA_plot.html')



# Report Main 에서 여러개의 Modeling을 분석한 후 결과를 적재
def Report(request):
    client_ip = get_client_ip(request)[0]

    df_refined = pd.read_csv(os.path.abspath(os.path.join(settings.DB_ROOT, '{}.csv'.format(client_ip))), low_memory=False)
    json_path = os.path.abspath(os.path.join(settings.DB_ROOT, '{}_meta-info.json'.format(client_ip)))
    with open(json_path, 'r') as f:
        json_info = json.load(f)


    #  Modeling & 결과 적재 ---------------------------------------------------------------------------------------
    name, mean_acc, mean_f1, confusion_final, confusion_labels, desc_name, desc_imp = XGB_n_Target(df_refined, json_info)

    # 2) Result 적재
    model_info = {}
    model_info[name] = {}
    model_info[name]['acc_score'] = mean_acc
    model_info[name]['f1_score'] = mean_f1
    model_info[name]['confusion_labels'] = confusion_labels
    model_info[name]['confusion_matrix'] = confusion_final
    model_info[name]['feature_name'] = desc_name
    model_info[name]['feature_imp'] = desc_imp

    # 전체 modeling 결과 저장
    # 추후에 model을 선택해서 저장하게 할 때 팔요
    with open(os.path.abspath(os.path.join(settings.DB_ROOT, '{}_model-info.json'.format(client_ip))),'w') as f: 
        json.dump(model_info, f) 

    # 버튼으로 나타낼 model 선택
    # <button>의 value값을 model명으로 해서 값을 Report_plot에 넘김
    model_zip = get_render_models(model_info)

    return render(request, './MachineLearning/Report_main.html', {'model_zip':model_zip})


def Report_plot(request):
    client_ip = get_client_ip(request)[0]

    # modeling 결과 창

    # Report_main에서 model명을 받아왔을 때
    if request.method == 'POST':
        model_json = os.path.abspath(os.path.join(settings.DB_ROOT, '{}_model-info.json'.format(client_ip)))
        with open(model_json, 'r') as f:
            model_info = json.load(f)

        # button value(model명) 정보를 이용하여 render할 값 뽑기
        model_name = request.POST.get('model-button')
        choice_model_info = model_info[model_name]

        # back → front 로 전달할 값
        # 1) Mean accuracy & F1_score
        mean_acc = round((choice_model_info['acc_score'])*100,1)
        mean_f1 = round((choice_model_info['f1_score'])*100,1)

        # 2) Confusion Matrix
        confusion_matrix = choice_model_info['confusion_matrix']
        confusion_labels = choice_model_info['confusion_labels']
        # confusion_labels의 경우 X_label순서와 Y_label순서가 반대
        confusion_labels_XY = [confusion_labels, confusion_labels[::-1]] # X_ls, Y_ls
        # heatmap의 textColor를 위해 confusion_matrix의 value값들의 평균 넘김
        confusion_mean = np.mean(sum(confusion_matrix,[]))

        # 3) feature importance
        # 20개까지 제한
        # Report_plot에서 bar chart를 그릴때에도 쓰임
        feature_name = choice_model_info['feature_name'][:20][::-1]
        feature_imp = choice_model_info['feature_imp'][:20][::-1]   

        # 4) 주요변수 distplot을 그리기 위한 값들
        # meta-info에서 target이 True인 값을 가져와서 y로 설정     
        db_csv = pd.read_csv(os.path.abspath(os.path.join(settings.DB_ROOT, '{}.csv'.format(client_ip))), low_memory=False)
        db_json = os.path.abspath(os.path.join(settings.DB_ROOT, '{}_meta-info.json'.format(client_ip)))
        with open(db_json, 'r') as f:
            json_info = json.load(f)

        # feature engineering할 때 NaN 처리 (median, mean, drop, ...)
        # Distplot을 나타낼 때는 NaN값은 제외하고 그리겠음
        # 일단은 target을 이런식으로 설정하지만 후에는 json_info에서 target == True인 값으로 설정해야함

        # ※ 뽑은 중요 변수들에는 category도 들어감 b/c target encoding을 사용하였기 때문
        # BUT 이러한 변수들은 distplot을 그릴 수 X
        # → stacked bar / side bar로 표현
        target = 'Survived'
        distplot_ls = []
        distplot_text = []
        for feature in feature_name:
            if judge_draw(json_info, db_csv, feature, target) == False:
                distplot_ls.append(['Impossible'])
                dist_text = '<div style="width:100%; height:120px"></div><div style="width:100%;"><span style="font-weight:600; font-size: 1.2em">그래프를 그릴 수 없습니다.</span></div><b><ol style="margin-left:-20px;margin-right:20px"><li style="line-height:2.3; font-size: 0.9em">선택한 변수의 모든 값이 Unique한 경우 (EX. ID, Name ..)</li><li style="line-height:2.3; font-size: 0.9em">선택한 변수의 모든 값이 Missing(NaN)인 경우 </li></ol></b>'
                distplot_text.append(dist_text)

            else:
                temp = db_csv[[feature,target]]
                temp = temp.dropna()
                temp.reset_index(drop=True, inplace=True)

                if json_info[feature]['selected_type'] == 'Numeric':
                    Distplots = distplot(json_info, temp, target, feature) # [label_num, final_labels, data_ls]
                    Distplots = ['Numeric'] + Distplots
                    distplot_ls.append(Distplots)

                    _, dist_text = graph_descrip('distplot', json_info, target, col2=feature)
                    distplot_text.append(dist_text)

                elif json_info[feature]['selected_type'] == 'Categorical':
                    barplots = side_stacked_chart(json_info, temp, feature, target) # [X_ls, Y_ls, label_num1, data_ls, mode]
                    barplots = ['Categorical'] + barplots
                    distplot_ls.append(barplots)

                    _, dist_text = graph_descrip('side-bar', json_info, feature, col2=target)
                    distplot_text.append(dist_text)

        # 5) 사용된 feature들에 관한 설명
        # meta-info에 있는 값들 (uniqueness, missing, ...) 가져오기
        # 일단은 col_types, uniqueness, missing 만!
        #   - 임시적으로 col_types를 넣음. 추후 변경 예정
        key_ls = [] ; col_ls = [] ; uni_ls = [] ; miss_ls = [] ; descrip_ls = []
        for key in json_info.keys():
            col_type, uni, miss = json_info[key]['selected_type'], json_info[key]['uniqueness'], json_info[key]['missing']
            key_ls.append(key) ; col_ls.append(col_type) ; uni_ls.append(uni) ; miss_ls.append(miss)

            data_col = db_csv[key]
            if col_type == 'Numeric':
                col_mean = round(np.mean(data_col),1); col_med = round(np.median(data_col),1)
                col_std = round(np.std(data_col),1); col_range = '{} - {}'.format(np.min(data_col), np.max(data_col));
                # desp = '{}는 숫자형 변수입니다. 해당 변수의 범위는 {}이며 평균은 {}, 표준편차는 {}입니다.'.format(key,col_range,col_mean,col_std)
                desp = '해당 변수의 범위는 {}이며 평균은 {}, 표준편차는 {}입니다.'.format(col_range,col_mean,col_std)

            elif col_type == 'Categorical':
                unique_num = len(data_col.unique()); most_freq = data_col.value_counts().idxmax();
                freq_num = data_col.value_counts().max();
                # desp = '{}는 범주형 변수입니다. 해당 변수는 총 {}개의 category로 이루어져 있으며, 그 중 {}이(가) {}으로 가장 많은 부분을 차지하고 있습니다.'.format(key,unique_num,most_freq,freq_num)
                desp = '해당 변수는 총 {}개의 category로 이루어져 있으며, 그 중 {}이(가) {}으로 가장 많은 부분을 차지하고 있습니다.'.format(unique_num,most_freq,freq_num)

            elif col_type == 'Time':
                col_range = '{} - {}'.format(np.min(data_col), np.max(data_col));
                # desp = '{}는 시간 변수입니다. 해당 변수는 {}사이의 시간을 나타냅니다. '.format(key,col_range)
                desp = '해당 변수는 {}사이의 시간을 나타냅니다. '.format(col_range)
            descrip_ls.append(desp)
        explain_zip = zip(key_ls, col_ls, uni_ls, miss_ls, descrip_ls)

        return render(request, './MachineLearning/Report_plot.html',{'mean_acc':[mean_acc, 100-mean_acc], 'mean_f1':[mean_f1,100-mean_f1],
                                                    'confusion_labels_XY':confusion_labels_XY, 'confusion_matrix':confusion_matrix, 'confusion_mean':confusion_mean,
                                                    'feature_name':[feat+'  ' for feat in feature_name], 'feature_imp':feature_imp,
                                                    'distplot_ls':distplot_ls, 'dist_text':distplot_text ,
                                                    'explain_zip':explain_zip,
                                                    'Target':target})

    # 맨 처음 창
    else:
        return render(request, './MachineLearning/Report_plot.html')
