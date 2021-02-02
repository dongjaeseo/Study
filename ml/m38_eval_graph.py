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
model.fit(x_train,y_train,verbose = 1, eval_metric=['rmse','logloss'], eval_set= [(x_train,y_train),(x_test,y_test)], early_stopping_rounds= 100)

aaa = model.score(x_test, y_test)
print('aaa : ', aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

result = model.evals_result()

import matplotlib.pyplot as plt

epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, result['validation_1']['rmse'], label = 'Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

# 주의! 회귀/이중/다중분류에 대하여 eval_metric 이 다르다!!

## metrics 에 리스트 형태로 여러개 넣을 수 있다!
### earlystopping!!