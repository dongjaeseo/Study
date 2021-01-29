# 실습
# RandomSearch, GS, Pipeline 을 엮어라
# 모델은 RandomForest

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

parameters = [
    {'randomforestclassifier__n_estimators' : [100,200,300], 'randomforestclassifier__min_samples_split' : [2,4], 'randomforestclassifier__max_depth' : [1,2,4], 'randomforestclassifier__max_leaf_nodes' : [1,2,4]}
]

scales = [MinMaxScaler(), StandardScaler()]
search = [RandomizedSearchCV, GridSearchCV]

for i in scales:
    pipe = make_pipeline(i, RandomForestClassifier())
    for j in search:
        model = j(pipe, parameters, cv = 5)
        model.fit(x_train,y_train)
        print(f'score_{i}_{j.__name__} : ', model.score(x_test,y_test))

# score_MinMaxScaler()_RandomizedSearchCV :  0.7
# score_MinMaxScaler()_GridSearchCV :  0.9333333333333333
# score_StandardScaler()_RandomizedSearchCV :  0.9333333333333333
# score_StandardScaler()_GridSearchCV :  0.9333333333333333