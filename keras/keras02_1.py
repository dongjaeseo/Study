import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras

from tensorflow.keras.layers import Dense


#1. data
x_train = np.array([1,2,3,4,5,10,11,12,13,14,15,16,17,18,19,20,21])
y_train = np.array([1,2,3,4,5,10,11,12,13,14,15,16,17,18,19,20,21])

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
model.compile(loss ='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

#4. evaluation, prediction
loss = model.evaluate(x_test, y_test, batch_size = 1)
print('loss: ', loss)

result = model.predict([9])
print("result: ", result)

# add layer / add train number / add node




