import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras

from tensorflow.keras.layers import Dense


#1. data
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_validation = np.array([6,7,8])
y_validation = np.array([6,7,8])

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

#2. model
model = Sequential()
model.add(Dense(5,input_dim = 1, activation='relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))


#3. compile, train
# model.compile(loss ='mse', optimizer = 'adam', metrics =['accuracy']) # metrics라는 지표?
# model.compile(loss ='mse', optimizer = 'adam', metrics =['mse'])
model.compile(loss ='mse', optimizer = 'adam', metrics =['mae']) # metrics에 대괄호는 리스트로 만들어주게끔 해서 나중에 두개이상 데이터를 넣을예정
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data=(x_validation,y_validation)) # 핏할때 검증용 데이터를 넣으면 검증도 같이한다

#4. evaluation, prediction
loss = model.evaluate(x_test, y_test, batch_size = 1)
print('loss: ', loss)

# result = model.predict([9])
result = model.predict([x_train])

print("result: ", result)