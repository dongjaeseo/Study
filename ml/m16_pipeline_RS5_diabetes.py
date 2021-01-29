# 실습
# RandomSearch, GS, Pipeline 을 엮어라
# 모델은 RandomForest

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

parameters = [
    {'randomforestregressor__n_estimators' : [100,200,300], 'randomforestregressor__min_samples_split' : [1,2,4], 'randomforestregressor__max_depth' : [1,2,4], 'randomforestregressor__max_leaf_nodes' : [1,2,4]}
]

scales = [MinMaxScaler(), StandardScaler()]
search = [RandomizedSearchCV, GridSearchCV]

for i in scales:
    pipe = make_pipeline(i, RandomForestRegressor())
    for j in search:
        model = j(pipe, parameters, cv = 5)
        model.fit(x_train,y_train)
        print(f'score_{i}_{j.__name__} : ', model.score(x_test,y_test))

# score_MinMaxScaler()_RandomizedSearchCV :  0.3064521503893538
# score_MinMaxScaler()_GridSearchCV :  0.30294539440924984
# score_StandardScaler()_RandomizedSearchCV :  0.30847208392605285
# score_StandardScaler()_GridSearchCV :  0.3057166220376121