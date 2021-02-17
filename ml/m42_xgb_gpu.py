import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor

# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x,y, shuffle = True, train_size = 0.8)

# from sklearn.preprocessing import MinMaxScaler
# scale = MinMaxScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

#2. modelling
model = XGBRegressor(n_estimators=100000, learning_rate=0.01,tree_method = 'gpu_hist', predictor='cpu_predictor', gpu_id=0)

#3. compile fit
model.fit(x_train,y_train, verbose = 1, eval_metric=['rmse'],eval_set = [(x_train,y_train),(x_test,y_test)], early_stopping_rounds=10000)

#4. evaluation, prediction
result = model.score(x_test,y_test)
print('model_score : ', result)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print('r2_score : ', r2)

# KNeighborsRegressor()
# model_score :  0.6351438014004257
# r2_score :  0.6351438014004257

# DecisionTreeRegressor()
# model_score :  0.7675966240423899
# r2_score :  0.7675966240423899

# RandomForestRegressor()
# model_score :  0.8920444822922753
# r2_score :  0.8920444822922753

# LinearRegression()
# model_score :  0.6710620404241625
# r2_score :  0.6710620404241625