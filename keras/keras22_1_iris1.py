import numpy as np
from sklearn.datasets import load_iris

# x,y = load_iris(return_X_y = True)  # :::: same as below

dataset = load_iris()
x = dataset.data
y = dataset.target

# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape)  # (150, 4)
# print(y.shape)  # (150,)

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x,y, shuffle = True, train_size = 0.8)

x_pred = x_test[-5:]
y_final = y_test[-5:]
y_test = y_test[:-5]
x_test = x_test[:-5]

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_pred = scale.transform(x_pred)

## onehot encoding
from tensorflow.keras.utils import to_categorical # convert 0,1,2 to [1,0,0] form

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_final = to_categorical(y_final)


#2. modelling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input = Input(shape = (4,))
d = Dense(100, activation = 'relu')(input)
d = Dense(100, activation = 'relu')(d)
d = Dense(100, activation = 'relu')(d)
d = Dense(100, activation = 'relu')(d)
d = Dense(3, activation = 'softmax')(d) # 노드의 수를 아웃풋의 수로 만들고 액티베이션 = 소프트맥스 (다중분류시)

model = Model(inputs = input, outputs = d)

#3. compile fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc','mae'])
model.fit(x_train,y_train, validation_split = 0.2, epochs = 100, verbose =2)

#4. evaluation, prediction

import tensorflow as tf

loss = model.evaluate(x_test,y_test)
print(loss)
y_pred = model.predict(x_pred)
y_dk = np.argmax(y_pred, axis = 1)
print(y_dk)
print(y_final)

# argmax search

# [0.030015692114830017, 1.0, 0.015097531490027905]