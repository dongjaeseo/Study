# keras21_cancer1.py 를 다중분류로 코딩하시오
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size = 0.8, shuffle = True)
x_test,x_pred,y_test,y_actual = tts(x_test,y_test,train_size = 0.6, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_pred = scale.transform(x_pred)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_actual = to_categorical(y_actual)

#2. modelling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100,activation = 'relu', input_dim = 30))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.summary()

#3. compile fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc', 'mae'])
model.fit(x_train,y_train,epochs = 100, verbose = 2, validation_split = 0.2)

#4. evaluation prediction
loss = model.evaluate(x_test,y_test)
print('accuracy : ', loss[1])

y_pred = model.predict(x_pred)
y_act = np.argmax(y_pred, axis = 1)
print(y_act)
print(y_actual)