import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('../data/npy/cifar10_x_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')


# plt.imshow(x_train[0])
# print(x_train.shape, y_train.shape) (50000, 32, 32, 3) (50000, 1)
# print(np.max(x_train),np.min(x_train),np.max(y_train),np.min(y_train)) 255 0   9 0

from sklearn.model_selection import train_test_split as tts
x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size = 0.8, shuffle = True, random_state = 0)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],3)/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],3)/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],3)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten, Input

input = Input(shape = (32,32,3))
d = Conv2D(100,2,padding = 'same', activation = 'relu')(input)
d = MaxPooling2D(2)(d)
d = Dropout(0.3)(d)
d = Conv2D(30,2, activation = 'relu')(d)
d = MaxPooling2D(2)(d)
d = Dropout(0.3)(d)
d = Flatten()(d)
d = Dense(10, activation = 'relu')(d)
d = Dropout(0.3)(d)
d = Dense(10, activation = 'softmax')(d)

model = Model(inputs = input, outputs = d)
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 20)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train,y_train,epochs = 1000, validation_data = (x_val,y_val), callbacks = [es], batch_size = 64)

#4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size = 64)
print('acc : ', loss[1])

# acc :  0.671500027179718
# acc :  0.6517000198364258
# acc :  0.6654999852180481

