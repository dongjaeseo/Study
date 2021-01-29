# 모델은 랜덤포레스트 사용
# 파이프라인 엮어서 25번 돌리기!!!
# 데이터는 wine 사용!!

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits = 5, shuffle = True)

parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10], 'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1]}
]


# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)

# pipe = make_pipeline(MinMaxScaler, model)

# score = cross_val_score(pipe, x, y, cv = kfold)

# print(score)