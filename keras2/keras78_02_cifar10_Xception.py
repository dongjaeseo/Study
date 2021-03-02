# 실습
# cifar10

import numpy as np

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, UpSampling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape, x_test.shape) (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3)
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델
vgg16 = Xception(include_top= False, weights = 'imagenet', input_shape = (96,96,3))
vgg16.trainable = False

model = Sequential()
model.add(UpSampling2D(size = (3,3)))
model.add(vgg16)
model.add(Flatten())
model.add(Dense(4096, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation= 'softmax'))

# model.summary()

#3. 컴파일 훈련
es = EarlyStopping(patience = 10)
lr = ReduceLROnPlateau(factor = 0.25, patience = 5, verbose = 1)
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

#####################################

# loss :  1.7066929340362549
# acc :  0.7878999710083008