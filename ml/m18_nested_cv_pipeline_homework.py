# 모델 : RandomForestClassifier



import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits = 5, shuffle = True)

parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10], 'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1]}
]

pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())

#2. modelling
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)

score = cross_val_score(model, x, y, cv = kfold) # 그리드서치에서 최적의 값이 나온걸 또 5번 돌려서

print('교차검증점수 : ', score)
# scores = cross_val_score(model, x_train, y_train, cv = kfold)

# 교차검증점수 :  [1.         0.95833333 0.875      0.95833333 0.95833333]