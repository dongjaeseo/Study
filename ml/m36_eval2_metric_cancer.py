# eval set
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x_train, x_test,y_train, y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, train_size = 0.8)

#2. 모델
model = XGBClassifier(n_jobs = 8, use_label_encoder= False, learning_rate = 0.01, n_estimators = 10)

#3. 훈련
model.fit(x_train,y_train, verbose= True, eval_metric = ['logloss','mae'], eval_set = [(x_train,y_train), (x_test,y_test)])

#4. 평가
y_pred = model.predict(x_test)
print('모델 점수는? : ', model.score(x_test,y_test))
print('정확도점수는? : ', accuracy_score(y_test,y_pred))

result = model.evals_result()
print('result : ', result)

# 모델 점수는? :  0.956140350877193
# 정확도점수는? :  0.956140350877193
# result :  {'validation_0': OrderedDict([('logloss', [0.684414, 0.675903, 0.667552, 0.659357, 0.651238,
#  0.64334, 0.635512, 0.627896, 0.620341, 0.612919])]), 'validation_1': OrderedDict([('logloss', [0.68447,
#   0.676096, 0.667886, 0.659773, 0.651487, 0.643736, 0.635805, 0.628279, 0.620596, 0.61299])])}

# 각각의 eval_set 에 대하여 eval_metric 점수가 n_estimators 만큼 반복!