import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

x = x.reshape(13,3,1)
x_pred = x_pred.reshape(1,3,1)


#2. 모델링

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM ,SimpleRNN, GRU

model = Sequential()
model.add(GRU(10, activation= 'relu', input_shape = (3,1)))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))

# model.summary()

#3. 컴파일 훈련
# from tensorflow.keras.callbacks import EarlyStopping
# ear = EarlyStopping(monitor = 'loss', patience= 10, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs = 200, batch_size = 1)

#4. 평가 예측
loss = model.evaluate(x,y,batch_size = 1)
print('loss : ', loss)
y_pred = model.predict(x_pred)
print(y_pred)

# LSTM
# loss :  0.09087680280208588
# [[79.78132]]
# [[80.0424]]
# [[79.88142]]

# SimpleRNN
# loss :  0.03615441918373108
# [[79.874664]]
# loss :  3.244553454351262e-06
# [[80.00449]]

# GRU
# loss :  0.4634828567504883
# [[81.69652]]
# loss :  0.013511969707906246
# [[80.226974]]

# 거의 유사하지만 성능이 조금씩 다르다 
# 통상적으로 LSTM 이 성능이 더 좋다고 판단

# GRU param# 이 다른이유: 반복되는 output에도 바이어스가 붙어서 >> (no.output+1  +  no.input+1) * (no.output) * 3 >> (10+1 + 1+1) * (10) * 3 
# >> 13*10*3 = 390