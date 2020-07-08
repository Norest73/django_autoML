import pandas as pd
import numpy as np
import os
import re
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
def XGB_n_Target(csv_file, refined_info, target):
	train = csv_file.copy()

	# My version
	# 변형한 upload파일 dtype을 원본으로 수정
	# 준형 매니저님이 사용하실 때는 해당 코드들 주석처리하고 사용하셔야 함
	for col in train.columns:
		if refined_info[col]['selected_type'] != 'Time':
			train[col] = train[col].astype(refined_info[col]['changed_dtype'])
		else:
			pass

	# target 설정
	target = target
	# target이 본래 numeric인데 object로 변형되었을 경우 원형인 numeric으로 되돌림
	# 만약 target이 본래도 object 였으면 LabelEncoder로 numeric 형식으로 만들어줌
	# → Change_label인 True인 경우에만(찐 object인 경우) numeric으로 변형한 confusion label 리스트를 다시 cateogry(string)로 바꿔줌 
	Change_label = False
	if refined_info[target]['selected_type'] == 'Categorical':
		if ('int' in refined_info[target]['changed_dtype']) | ('float' in refined_info[target]['changed_dtype']):
			train[target] = train[target].astype(refined_info[target]['changed_dtype'])
			confusion_labels = list(set(train[target])) # 알아서 sorted
		else:
			# 찐 object
			# LabelEncoder를 하지 않은 상태의 unique category
			ori_col_unique = train[target].unique()
			encoder = LabelEncoder()
			encoder.fit(train[target])
			train[target] = encoder.transform(train[target])

			# LabelEncoder를 한 상태의 unique category
			encoded_col_unique = train[target].unique()
			# confusion matrix에서 정렬할 순서
			encoder_order = encoded_col_unique.argsort()
			confusion_labels = encoded_col_unique[encoder_order].tolist()
			# encoder_order대로 original_label을 정렬
			ori_ordered_labels = ori_col_unique[encoder_order].tolist()
			Change_label = True
	else:
	    confusion_labels = list(set(train[target]))

	# object인 columns
	# target encoding을 적용할 컬럼들
	object_cols = [key for key in refined_info.keys() if ('object' in refined_info[key]['changed_dtype'])&('Time' not in refined_info[key]['selected_type'])]
	not_object = [key for key in refined_info.keys() if (key not in object_cols)&('Time' not in refined_info[key]['selected_type'])] 

	if target in object_cols:
	    object_cols.remove(target)
	elif target in not_object:
	    not_object.remove(target)

	y = train[target]
	X = train.drop(target,axis=1)

	data_skf = StratifiedKFold(n_splits=5, random_state=42)
	encoding_skf = StratifiedKFold(n_splits=5, random_state=42)

	acc_ls = []
	f1_ls = []
	confusion_ls = []
	imp_ls = []

	# train data와 test data 나누기 (by cross validation)
	# 이때 train 데이터의 경우 한번 더 cross validation을 통해 target encoding 시킴

	for train_index, test_index in data_skf.split(X,y):
	    X_train, y_train = X.iloc[train_index,:].reset_index(drop=True,inplace=False), y[train_index].reset_index(drop=True,inplace=False)
	    X_val, y_val = X.iloc[test_index,:].reset_index(drop=True,inplace=False), y[test_index].reset_index(drop=True,inplace=False)
	    
	    # new: target encoding시킬 X_train + y_train 데이터
	    # modeling train에 쓰일 데이터 생성
	    new = train.iloc[train_index,:].reset_index(drop=True,inplace=False)

	    for col in object_cols:
	        new['{}_mean'.format(col)] = np.nan
	    
	    # cross validation을 통해 target_encoding된 훈련데이터를 만드는 과정
	    for encoding_index, encoded_index in encoding_skf.split(X_train,y_train):
	        X_encoding = train.iloc[encoding_index,:]
	        X_encoded = train.iloc[encoded_index,:]

	        for col in object_cols:
	            # 방법1) 일반 target encoding 
	            object_mean = X_encoded[col].map(X_encoding.groupby(col)[target].mean())
	            X_encoded['{}_mean'.format(col)] = object_mean
	            new.iloc[encoded_index] = X_encoded
	            
	            # 방법2) smoothing
	#             global_mean = X_encoding[target].mean()
	#             alpha = 2
	            
	#             n_size = X_encoded[col].map(X_encoding.groupby(col).size())
	#             target_mean = X_encoded[col].map(X_encoding.groupby(col)[target].mean())
	#             X_encoded[col] = (target_mean*n_size + global_mean*alpha) / (n_size + alpha)
	#         new.iloc[encoded_index] = X_encoded

	    global_mean = train[target].mean()
	    for col in object_cols:
	        new['{}_mean'.format(col)].fillna(global_mean, inplace=True)
	    
	    # not_obect cols + obeject_mean cols
	    model_cols = ['{}_mean'.format(col) for col in object_cols] + not_object + [target]
	    model_train, train_target = new[model_cols].drop(target,axis=1), new[target]
	    model_train.columns = [col if '_mean' not in col else col.replace('_mean','') for col in model_train.columns]
	    
	    # for validation
	    # modeling validation에 쓰일 데이터 생성
	    model_val, val_target = X_val[object_cols + not_object], y_val
	    for col in object_cols:
	        model_val[col] = model_val[col].map(new.groupby(col)['{}_mean'.format(col)].mean())
	        model_val[col].fillna(new['{}_mean'.format(col)].mean(),inplace=True)
	    
	    model = XGBClassifier(random_state=42)
	    model.fit(model_train, train_target)
	    y_predict = model.predict(model_val)

	    acc = accuracy_score(val_target, y_predict)
	    acc_ls.append(acc)

	    f1 = f1_score(val_target, y_predict)
	    f1_ls.append(f1)

	    confusion = confusion_matrix(val_target, y_predict, labels=confusion_labels)
	    confusion_ls.append(confusion)

	    # feature importance
	    imp = model.feature_importances_
	    imp_ls.append(imp)

	mean_acc = np.mean(acc_ls)
	mean_f1 = np.mean(f1_ls)
	mean_imp = (np.mean(imp_ls, axis=0))

	# np.array 형식은 json으로 저장 불가능
	# 펴서 list의 형식으로 저장
	# 나중에 render할 때 사용하기 편하게 confusion_matrix의 아랫단부터 값 제공
	where = f1_ls.index(np.max(f1_ls))
	confusion_final = confusion_ls[where]
	confusion_final = confusion_final.tolist()[::-1]
	if Change_label == True:
		confusion_labels = ori_ordered_labels

	# dataframe 형식은 json으로 저장 불가능
	# importance와 name을 따로 저장 (descending 순서로 → asecending = False)
	imp_order = mean_imp.argsort()[::-1]
	desc_imp = mean_imp[imp_order].tolist()
	desc_name = np.array([col for col in model_train.columns])[imp_order].tolist()

	name = 'XGB{}'.format(random.random()).replace('0.','')

	return name, mean_acc, mean_f1, confusion_final, confusion_labels, desc_name, desc_imp
	    

# render할 model 선택
# top 변수를 통해 몇개의 모델까지 render 시킬지 결정
# 0610: render할 모델의 선정 기준 필요 → 일단은 f1_score 기준으로 top 뽑는 형식으로
def get_render_models(model_info, top=1):
	model_names = np.array([key for key in model_info.keys()])
	f1_ls = np.array([model_info[key]['f1_score'] for key in model_info.keys()])
	f1_order = f1_ls.argsort()[::-1]

	f1_model_names = model_names[f1_order].tolist()
	final_model_names = f1_model_names[:top]
	model_nums = ['Model({})'.format(i) for i in range(1,top+1)]
	model_zip = zip(final_model_names, model_nums)

	return model_zip

