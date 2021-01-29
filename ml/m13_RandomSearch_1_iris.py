# 모델 : RandomForestClassifier


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# dataset = load_iris()
# x = dataset.data
# y = dataset.target

dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header = 0, index_col = 0)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test, y_train,y_test = train_test_split(x,y,train_size = 0.8)
kfold = KFold(n_splits = 5, shuffle = True)
RandomForestClassifier()
# parameters = [
#     {'n_estimators' : [100,200]},
#     {'max_depth' : [6,8,10,12]},
#     {'min_samples_leaf' : [3,5,7,10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1]}
# ] ## SVC 에 들어가는 파라미터를 조정해주는것

parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10], 'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1]}
]
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=10,
#                        n_jobs=-1)
# 최종정답률 0.9333333333333333
# 최종정답률 0.9333333333333333

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(x)
x = scale.transform(x)
#2. modelling
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)
# scores = cross_val_score(model, x_train, y_train, cv = kfold)

# #3. compile fit
model.fit(x_train,y_train)

# #4. evaluation, prediction
print('최적의 매개변수 : ', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률', accuracy_score(y_test,y_pred))

# 최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_leaf=3, min_samples_split=10,
#                        n_jobs=-1)
# 최종정답률 0.9666666666666667