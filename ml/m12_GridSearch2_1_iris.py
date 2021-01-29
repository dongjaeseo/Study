# 모델 : RandomForestClassifier



import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

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
import datetime
t1 = datetime.datetime.now()
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold)
model.fit(x_train,y_train)
t2 = datetime.datetime.now()
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)
model.fit(x_train,y_train)
t3 = datetime.datetime.now()
# scores = cross_val_score(model, x_train, y_train, cv = kfold)

# #3. compile fit
print('Grid 걸린 시간 : ', t2-t1)
print('Random 걸린 시간 : ', t3-t2)


# # #4. evaluation, prediction
# print('최적의 매개변수 : ', model.best_estimator_)

# y_pred = model.predict(x_test)
# print('최종정답률', accuracy_score(y_test,y_pred))

# Grid 걸린 시간 :  0:01:27.178281
# Random 걸린 시간 :  0:00:06.405835