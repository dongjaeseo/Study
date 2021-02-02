# eval set
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y = True)
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state = 33)

#2. 모델
model = XGBRegressor(n_estimators = 10000, learning_rate = 0.01, j_nobs = 8)

#3. 훈련
model.fit(x_train,y_train,verbose = 1, eval_metric=['rmse','logloss', 'mae'], eval_set= [(x_train,y_train),(x_test,y_test)], early_stopping_rounds= 10)

aaa = model.score(x_test, y_test)
print('aaa : ', aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

# result = model.evals_result()
# print(result)

import pickle
# pickle.dump(model, open('../data/xgb_save/m39.pickle.dat', 'wb'))
# print('저장완료')

# 주의! 회귀/이중/다중분류에 대하여 eval_metric 이 다르다!!

## metrics 에 리스트 형태로 여러개 넣을 수 있다!
### earlystopping!!
model2 = pickle.load(open('../data/xgb_save/m39.pickle.dat', 'rb'))
print('불러왔다!')
r22 = model2.score(x_test,y_test)
print('r22 : ', r22)