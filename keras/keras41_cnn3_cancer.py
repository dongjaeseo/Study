import numpy as np

#1. 데이터
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
# print(x.shape,y.shape) (569, 30) (569,)

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x,y,train_size = 0.8, shuffle = True, random_state = 1)
x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size = 0.8, shuffle = True, random_state = 1)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],2,5,3)
x_test = x_test.reshape(x_test.shape[0],2,5,3)
x_val = x_val.reshape(x_val.shape[0],2,5,3)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Dropout, Flatten

input = Input(shape = (2,5,3))
d = Conv2D(100, (2,3), activation ='relu', padding = 'same')(input)
d = Dropout(0.3)(d)
d = Conv2D(100, 2)(d)
d = Dropout(0.2)(d)
d = Flatten()(d)
d = Dense(30)(d)
d = Dense(1, activation = 'sigmoid')(d)

model = Model(inputs = input, outputs = d)
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auto')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train,y_train,epochs = 1000, validation_data=(x_val,y_val), callbacks = [es], batch_size = 1)

#4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

# loss :  0.0636897161602974
# acc :  0.9736841917037964

# loss :  0.09301544725894928
# acc :  0.9824561476707458