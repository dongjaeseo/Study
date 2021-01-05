# keras23_LSTM_scale DNN으로 코딩

# DNN으로 23번파일보다 loss를 좋게 만들것


import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (100,) 형태 혹은 (1,100) 둘다 되네??
x_pred = np.array([[50,60,70]])

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x)
x = scale.transform(x)
# x_test = scale.transform(x_test)
# x_val = scale.transform(x_val)
x_pred = scale.transform(x_pred)

#2. 모델링

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(100, activation= 'relu', input_dim = 3))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))

# model.summary()

#3. 컴파일 훈련
# from tensorflow.keras.callbacks import EarlyStopping
# ear = EarlyStopping(monitor = 'loss', patience= 10, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs = 600, batch_size = 1, verbose =2)

#4. 평가 예측
loss = model.evaluate(x,y,batch_size = 1)
y_pred = model.predict(x_pred)
print('loss : ', loss)
print(y_pred)

# LSTM
# [[80.0424]]
# [[79.88142]]

# DNN
# loss :  0.007085837423801422
# [[79.973465]]
# loss :  0.0006358188111335039
# [[80.006775]