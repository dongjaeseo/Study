import numpy as np
import pandas as pd
from samsung_merge_function import size

#1. 데이터
x_train = np.load('./samsung/samsung_new_%d.npy'%size,allow_pickle=True)[0]
x_test = np.load('./samsung/samsung_new_%d.npy'%size,allow_pickle=True)[1]
x_val = np.load('./samsung/samsung_new_%d.npy'%size,allow_pickle=True)[2]
y_train = np.load('./samsung/samsung_new_%d.npy'%size,allow_pickle=True)[3]
y_test = np.load('./samsung/samsung_new_%d.npy'%size,allow_pickle=True)[4]
y_val = np.load('./samsung/samsung_new_%d.npy'%size,allow_pickle=True)[5]
x_pred = np.load('./samsung/samsung_new_%d.npy'%size,allow_pickle=True)[6]

#2. 모델
from tensorflow.keras.models import load_model
model = load_model('./samsung/samsungjuga_0115.hdf5')

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 8)
print('loss, mae : ', result)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
print('r2 : ', r2_score(y_test,y_pred))

y_next = model.predict(x_pred)
print(y_next)

##############################

