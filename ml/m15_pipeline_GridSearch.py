import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline # 둘이 거의 동일 쓰는법만 다르다

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header = 0, index_col = 0)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test, y_train,y_test = train_test_split(x,y,train_size = 0.8)

# parameters = [
#     {'randomforestclassifier__n_estimators' : [100,200], 'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10], 'randomforestclassifier__min_samples_split' : [2, 3, 5, 10], 'randomforestclassifier__n_jobs' : [-1]}
# ]

parameters = [
    {'mal__n_estimators' : [100,200], 'mal__max_depth' : [6,8,10,12], 'mal__min_samples_leaf' : [3,5,7,10], 'mal__min_samples_split' : [2, 3, 5, 10], 'mal__n_jobs' : [-1]}
]

# 모델     pipe : 전처리까지 합친다
pipe = Pipeline([('scale',MinMaxScaler()),('mal',RandomForestClassifier())]) ## 얘를 모델화 시켜서 estimator 부분에 넣어줌
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

model = RandomizedSearchCV(pipe, parameters, cv = 5)

model.fit(x_train,y_train)
result = model.score(x_test,y_test)
print(result)