import numpy as np

#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])
x_pred = ([[6,7,8]])

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x)
x = scale.transform(x)
x_pred = scale.transform(x_pred)  # 3차원 텐서로 바꾸기전에 전처리를 할 것

x = x.reshape(4,3,1)
x_pred = x_pred.reshape(1,3,1)    # 3차원으로 바꿔준다

#2. 모델링 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
input = Input(shape = (3,1))      # 3차원 텐서의 열, 잘라내는 수를 넣는다
d = LSTM(100, activation = 'relu')(input)
d = Dense(200)(d)
d = Dense(40)(d)
d = Dense(40)(d)
d = Dense(40)(d)
d = Dense(20)(d)
d = Dense(1)(d)

model = Model(inputs = input, outputs = d)
# model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs = 200, batch_size = 1, verbose = 2)

#4. 훈련 평가
loss = model.evaluate(x,y,batch_size = 1)
print('loss : ', loss)
y_pred = model.predict(x_pred)
print(y_pred)