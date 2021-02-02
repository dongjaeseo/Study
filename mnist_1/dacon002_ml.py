import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

train = pd.read_csv('./mnist_1/train.csv')
test = pd.read_csv('./mnist_1/test.csv')
submission = pd.read_csv('./mnist_1/submission.csv', header = 0)

x_train = train.loc[:, '0':].to_numpy()/255.
y_train = train.loc[:, 'digit'].to_numpy()
x_test = test.loc[:, '0':].to_numpy()/255.

# print(x_train.shape,x_test.shape,y_train.shape) (2048, 784) (20480, 784) (2048,)

parameters = [
    {'n_estimators' : [200], 'max_depth' : [2,4,6], 'n_jobs': [8],'learning_rate':[0.01, 0.05, 0.001]}
]

model = RandomizedSearchCV(XGBClassifier(), parameters, cv = 5)
model.fit(x_train,y_train, eval_metric = ['mlogloss'], verbose = 1, eval_set = [(x_train,y_train)], early_stopping_rounds = 15)

y_test = model.predict(x_test)

submission.iloc[:,1] = y_test

submission.to_csv('./mnist_1/submission1.csv', index = 0)


