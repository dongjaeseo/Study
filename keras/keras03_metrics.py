import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras

from tensorflow.keras.layers import Dense


#1. data
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

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
# model.compile(loss ='mse', optimizer = 'adam', metrics =['accuracy']) # metrics라는 지표를 써주면 정확도가 추가됨
# model.compile(loss ='mse', optimizer = 'adam', metrics =['mse'])
model.compile(loss ='mse', optimizer = 'adam', metrics =['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

#4. evaluation, prediction
loss = model.evaluate(x_test, y_test, batch_size = 1)
print('loss: ', loss)

# result = model.predict([9])
result = model.predict([x_train])

print("result: ", result)

# add layer / add train number / add node




