# 실습
# cifar10

import numpy as np

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape, x_test.shape) (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3)
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_val = preprocess_input(x_val)
# x_train = (x_train)/255.
# x_test = (x_test)/255.
# x_val = (x_val)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델
vgg16 = VGG19(include_top= False, weights = 'imagenet', input_shape = (32,32,3))
vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation= 'softmax'))

model.summary()

#3. 컴파일 훈련
es = EarlyStopping(patience = 21)
lr = ReduceLROnPlateau(factor = 0.25, patience = 7, verbose = 1)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 32, epochs = 10000, validation_data = (x_val, y_val), callbacks = [es, lr])

#4. 평가 예측
loss = model.evaluate(x_test, y_test, batch_size = 32)
print('loss : ', loss[0])
print('acc : ', loss[1])

# loss :  1.7287994623184204
# acc :  0.6352999806404114

# loss :  2.8665335178375244
# acc :  0.6714000105857849

#############################################

# loss :  3.0632424354553223
# acc :  0.6657000184059143