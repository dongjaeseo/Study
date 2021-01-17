import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data

x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')

x_test = x[-5:]
x = x[:-5]
y_test = y[-5:]
y = y[:-5]


# print(x.shape)    # (569,30)
# print(y.shape)    # (569,)
# print(x[:5])
# print(y)

# preprocess / MinMaxScaler / tts

#2. modelling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, activation ='relu', input_shape=(30,)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3. compile fit
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics = ['acc']) # for binary classification, use loss as cross entropy
# model.compile(loss='mean_squared_error',optimizer = 'adam', metrics = ['acc'])
model.fit(x,y, epochs = 1500, validation_split = 0.2, verbose = 2, callbacks = [early])

loss = model.evaluate(x,y)
print(loss)
# print(y_test[-5:])

# 1. acc > 0.985
# 2. print predict for y[0:5]
y_pred = model.predict(x_test)
print(y_test)
# for number in range(len(y_pred)):
#     if y_pred[number] < 0.5 :
#         y_pred[number] = 0
#     else :
#         y_pred[number] = 1
#     number += 1
# print(y_pred.reshape(1,5))
y_pred = y_pred.reshape(5,)
y_pred = np.where(y_pred<0.5,0,1)
print(y_pred)
# [0.04216127097606659, 0.9894551634788513]
