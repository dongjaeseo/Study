import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([[50,60,70]])

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x)
x = scale.transform(x)
x_pred = scale.transform(x_pred)

x = x.reshape(13,3,1)
x_pred = x_pred.reshape(1,3,1)


#2. 모델링

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, activation= 'relu', input_shape = (3,1)))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))

# model.summary()

#3. 컴파일 훈련
# from tensorflow.keras.callbacks import EarlyStopping
# ear = EarlyStopping(monitor = 'loss', patience= 10, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs = 300, batch_size = 1)

#4. 평가 예측
y_pred = model.predict(x_pred)
print(y_pred)

# [[80.0424]]
# [[79.88142]]
