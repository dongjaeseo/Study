import numpy as np
import pandas as pd

#1. 데이터
x_train = np.load('./samsung/samsung.npz')['x_train']
x_test = np.load('./samsung/samsung.npz')['x_test']
x_val = np.load('./samsung/samsung.npz')['x_val']
x_pred = np.load('./samsung/samsung.npz')['x_pred']
y_train = np.load('./samsung/samsung.npz')['y_train']
y_test = np.load('./samsung/samsung.npz')['y_test']
y_val = np.load('./samsung/samsung.npz')['y_val']

#2. 모델
from tensorflow.keras.models import load_model

model = load_model('./samsung/samsungjuga.hdf5')

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 1)
print('loss, mae : ', result)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
print('r2 : ', r2_score(y_test,y_pred))

y_next = model.predict(x_pred)
print(y_next)

# 전후 데이터 다 쓴모델
# r2 :  0.9937017091301253
# [[88493.24]]
# loss, mae :  [1166993.375, 766.7100830078125]
# r2 :  0.9933660531669145
# [[91902.34]]
# loss, mae :  [1181757.625, 841.8319091796875]
# r2 :  0.9936705864431818
# [[91872.766]]
# loss, mae :  [2145229.75, 1073.52099609375]
# r2 :  0.9886229686283687
# [[91100.86]]
