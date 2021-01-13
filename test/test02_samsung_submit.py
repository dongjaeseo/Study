import numpy as np
import pandas as pd

#1. 데이터
x_train = np.load('../data/npy/ss_x_train.npy')
x_test = np.load('../data/npy/ss_x_test.npy')
x_val = np.load('../data/npy/ss_x_val.npy')
x_pred = np.load('../data/npy/ss_x_pred.npy')
y_train = np.load('../data/npy/ss_y_train.npy')
y_test = np.load('../data/npy/ss_y_test.npy')
y_val = np.load('../data/npy/ss_y_val.npy')

#2. 모델
from tensorflow.keras.models import load_model

model = load_model('../data/modelcheckpoint/samsungjuga_190-3634096.hdf5')

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 8)
print('loss, mae : ', result)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
print('r2 : ', r2_score(y_test,y_pred))

y_next = model.predict(x_pred)
print(y_next)

# loss, mae :  [1278777.0, 902.4755859375]
# r2 :  0.9794304449840759
# [[96138.62]]