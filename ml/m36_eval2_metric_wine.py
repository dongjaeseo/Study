# eval set
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x_train, x_test,y_train, y_test = train_test_split(load_wine().data, load_wine().target, train_size = 0.8)

#2. 모델
model = XGBClassifier(n_jobs = 8, use_label_encoder= False, learning_rate = 0.01, n_estimators = 10)

#3. 훈련
model.fit(x_train,y_train, verbose= True, eval_metric = ['mlogloss','merror'], eval_set = [(x_train,y_train), (x_test,y_test)])

#4. 평가
y_pred = model.predict(x_test)
print('모델 점수는? : ', model.score(x_test,y_test))
print('정확도점수는? : ', accuracy_score(y_test,y_pred))

# result = model.evals_result()
# print('result : ', result)

# 모델 점수는? :  0.9722222222222222
# 정확도점수는? :  0.9722222222222222
# result :  {'validation_0': OrderedDict([('mlogloss', [1.085045, 1.071708, 1.058596, 1.045702, 1.033022, 1.02055
#     , 1.008281, 0.996211, 0.984334, 0.972646])]), 'validation_1': OrderedDict([('mlogloss', [1.085674, 1.072896
#         , 1.06034, 1.048068, 1.036006, 1.024085, 1.012428, 1.000903, 0.989633, 0.978485])])}

# 각각의 eval_set 에 대하여 eval_metric 점수가 n_estimators 만큼 반복!