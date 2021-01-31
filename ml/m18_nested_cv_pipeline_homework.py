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
    {'randomforestclassifier__n_estimators' : [100,200], 'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10], 'randomforestclassifier__min_samples_split' : [2, 3, 5, 10]}
]


#2. modelling
pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())
model = RandomizedSearchCV(pipe, parameters, cv = kfold)
model.fit(x,y)

pipe2 = make_pipeline(MinMaxScaler(),model)
score = cross_val_score(model, x, y, cv = kfold) # 그리드서치에서 최적의 값이 나온걸 또 5번 돌려서

print('교차검증점수 : ', score)

# 교차검증점수 :  [1.         0.96666667 0.93333333 0.93333333 1.        ]
# 뭔가 했는데 맞는건지 모르겠다!!
# 파이프라인이 그냥 모델에 적용하기전 스케일을 해준다고 가정하고
# 먼저 랜덤서치 하기전에 파이프라인을 만들어서 스케일된 x에 최적인 모델을 찾아 model 에 저장했다!

# 두번째 파이프라인은 찾은 최적의 모델+스케일을 해서 교차검증하였다!!