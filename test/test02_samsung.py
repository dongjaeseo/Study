import numpy as np
import pandas as pd

#1. 데이터
x_train = np.load('../data/npy/ss_x_train.npy')
x_test = np.load('../data/npy/ss_x_test.npy')
x_val = np.load('../data/npy/ss_x_val.npy')
x_pred = np.load('../data/npy/ss_x_pred.npy') # 01.13 일의 데이터, 01.14 의 종가를 예측하기 위함

y_train = np.load('../data/npy/ss_y_train.npy')
y_test = np.load('../data/npy/ss_y_test.npy')
y_val = np.load('../data/npy/ss_y_val.npy')

#2. 모델
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM

model = load_model('../data/modelcheckpoint/samsungjuga_ 47.hdf5')

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 8)
print('loss, mae : ', result)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
print('r2 : ', r2_score(y_test,y_pred))

y_pred2 = model.predict(x_pred)
print(y_pred2)

# loss, mae :  [2562125.5, 1141.66455078125]
# r2 :  0.9661172643063053
# [[91059.47]]