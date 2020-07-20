import pandas as pd
import numpy as np
import os
import re
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import random
import warnings
# for modeling demo (0602)

# csv_file 부분에는 pandas dataframe 자체가 들어옴
# Adv Modeling (2)
# 	- Target encoding (Mean encoding) 적용
# 	- 기존의 pandas type이 object였던 변수만 Target encoding 적용
# confusion matrix에서 쓰일 label 역시 출력
def XGB_n_Target(df, json_info):

	# DataFrame Columns Data type 변환
	for col in df.columns:
		df[col] = df[col].astype(json_info[col]['changed_dtype'])

	# Target Feature 입력
	for col in json_info:
		if json_info[col]['target'] == True :
			target = col

	# Target Feature Encoding : 데이터 값을 숫자 형태로 변환
	# Target Feature는 Categorical 변수라고 가정 (Classification 모델을 우선적으로 개발)
	# Object 데이터 값을 Numeric으로 변형한 경우 confusion label 리스트를 다시 cateogry(str)로 바꿔줌 
	if 'Numeric' in json_info[target]['possible_type']:	
		confusion_labels = list(set(df[target]))  # set() : Sorted Unique Value in DF Series 

	else:  # Object(str) 형태의 데이터 값이 포함되어 있는 경우 숫자 형태로 변환

		ori_col_unique = df[target].unique()  # LabelEncoder를 하지 않은 String Unique Value List
		encoder = LabelEncoder()
		encoder.fit(df[target])
		df[target] = encoder.transform(df[target])

		# LabelEncoder를 한 상태의 unique category
		encoded_col_unique = df[target].unique()
		# confusion matrix에서 정렬할 순서
		encoder_order = encoded_col_unique.argsort()
		confusion_labels = encoded_col_unique[encoder_order].tolist()
		# encoder_order대로 original_label을 정렬
		ori_ordered_labels = ori_col_unique[encoder_order].tolist()

	df[target] = df[target].astype('int64')  # Target Feature가 int 변수일때만 모델링 가능

	# Target 숫자형/범주형 변수 구분 (범주형 변수에는 Target Encoding 적용) (아래로 이동 예정)
	category_cols = [i for i in json_info if 'object' == json_info[i]['changed_dtype'] and json_info[i]['selected_type'] != 'Time' and i != target]
	numeric_cols = [i for i in json_info if i not in category_cols and json_info[i]['selected_type'] != 'Time' and i != target] 

	df_X = df.drop(target, axis=1)
	df_Y = df[target]

	data_skf = StratifiedKFold(n_splits=5, random_state=42)
	encoding_skf = StratifiedKFold(n_splits=5, random_state=42)

	acc_ls = []
	pre_ls = []
	rec_ls = []
	f1_ls = []
	confusion_ls = []
	imp_ls = []

	# train data와 val data 나누기 (by cross validation)
	# 이때 train 데이터의 경우 한번 더 cross validation을 통해 target encoding 시킴

	for train_index, val_index in data_skf.split(df_X, df_Y):
	    X_train = df_X.iloc[train_index].reset_index(drop=True,inplace=False)
	    Y_train = df_Y[train_index].reset_index(drop=True,inplace=False)
	    X_val = df_X.iloc[val_index].reset_index(drop=True,inplace=False)
	    Y_val = df_Y[val_index].reset_index(drop=True,inplace=False)

	    # new: target encoding시킬 X_train + Y_train 데이터
	    # modeling train에 쓰일 데이터 생성
	    new = df.iloc[train_index,:].reset_index(drop=True,inplace=False)

	    for col in category_cols:
	        new['{}_mean'.format(col)] = np.nan
	    
	    # cross validation을 통해 target_encoding된 훈련데이터를 만드는 과정
	    for encoding_index, encoded_index in encoding_skf.split(X_train, Y_train):
	        X_encoding = new.iloc[encoding_index,:]
	        X_encoded = new.iloc[encoded_index,:]

	        for col in category_cols:
	            # 일반 target encoding 
	            object_mean = X_encoded[col].map(X_encoding.groupby(col)[target].mean()) # col 값에 대한 
	            X_encoded['{}_mean'.format(col)] = object_mean
	            new.iloc[encoded_index] = X_encoded

	    global_mean = df[target].mean()
	    for col in category_cols:
	        new['{}_mean'.format(col)].fillna(global_mean, inplace=True)
	    
	    # not_obect cols + obeject_mean cols
	    model_cols = ['{}_mean'.format(col) for col in category_cols] + numeric_cols + [target]
	    model_train, train_target = new[model_cols].drop(target,axis=1), new[target]
	    model_train.columns = [col if '_mean' not in col else col.replace('_mean','') for col in model_train.columns]
	    
	    # for validation
	    # modeling validation에 쓰일 데이터 생성
	    model_val, val_target = X_val[category_cols + numeric_cols], Y_val
	    for col in category_cols:
	        model_val[col] = model_val[col].map(new.groupby(col)['{}_mean'.format(col)].mean())
	        model_val[col].fillna(new['{}_mean'.format(col)].mean(),inplace=True)
	    
	    model = XGBClassifier(random_state=42)
	    model.fit(model_train, train_target)
	    Y_predict = model.predict(model_val)

	    acc = accuracy_score(val_target, Y_predict)
	    acc_ls.append(acc)

	    pre = precision_score(val_target, Y_predict)
	    pre_ls.append(pre)

	    rec = recall_score(val_target, Y_predict)
	    rec_ls.append(rec)

	    f1 = f1_score(val_target, Y_predict)
	    f1_ls.append(f1)

	    confusion = confusion_matrix(val_target, Y_predict, labels=confusion_labels)
	    confusion_ls.append(confusion)

	    # feature importance
	    imp = model.feature_importances_
	    imp_ls.append(imp)

	mean_acc = np.mean(acc_ls)
	mean_pre = np.mean(pre_ls)
	mean_rec = np.mean(rec_ls)
	mean_f1 = np.mean(f1_ls)
	mean_imp = (np.mean(imp_ls, axis=0))
	
	# F1 score가 가장 높은 값의 Confusion Matrix를 List 형태로 변환 (나중에 사용하기 편하게 아랫단부터 값 제공)
	where = f1_ls.index(np.max(f1_ls))
	confusion_final = confusion_ls[where]
	confusion_final = confusion_final.tolist()[::-1]

	# Target Feature Label Encoding 한 경우 원래 값으로 변경 
	if 'Numeric' not in json_info[target]['possible_type']:
		confusion_labels = ori_ordered_labels

	# Feature importance가 높은(descending) 순서대로 List 형태로 변환
	imp_order = mean_imp.argsort()[::-1]
	desc_imp = mean_imp[imp_order].tolist()
	desc_name = np.array([col for col in model_train.columns])[imp_order].tolist()

	name = 'XGB{}'.format(round(random.random(), 8)).replace('0.','')

	return name, mean_acc, mean_pre, mean_rec, mean_f1, confusion_final, confusion_labels, desc_name, desc_imp
	    
def get_render_models(model_info, top=1):
	''' F1 Score를 기준으로 Rendering Model 선별 '''

	model_names = np.array([key for key in model_info.keys()])
	f1_ls = np.array([model_info[key]['f1_score'] for key in model_info.keys()])
	f1_order = f1_ls.argsort()[::-1]

	f1_model_names = model_names[f1_order].tolist()
	final_model_names = f1_model_names[:top]
	model_nums = ['Model {}'.format(i) for i in range(1,top+1)]
	model_zip = zip(final_model_names, model_nums)

	return model_zip