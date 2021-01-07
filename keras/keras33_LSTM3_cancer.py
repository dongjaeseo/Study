import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# print(x.shape)    # (569,30)
# print(y.shape)    # (569,)

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size = 0.8)
x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size = 0.8)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2. modelling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, activation ='relu', input_shape=(30,1)))
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
model.fit(x_train,y_train, epochs = 1500, validation_split = 0.2, verbose = 2, callbacks = [early])

loss = model.evaluate(x_train,y_train)
print(loss)
y_pred = model.predict(x_test)
# for number in range(len(y_pred)):
#     if y_pred[number] < 0.5 :
#         y_pred[number] = 0
#     else :
#         y_pred[number] = 1
#     number += 1
# print(y_pred.reshape(1,5))

y_pred = y_pred.reshape(y_pred.shape[0],)
y_pred = np.where(y_pred<0.5,0,1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred,y_test)
print("acc_score : ", acc)

# acc_score :  0.956140350877193