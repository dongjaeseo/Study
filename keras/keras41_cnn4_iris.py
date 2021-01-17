import numpy as np

#1. 데이터
from sklearn.datasets import load_iris
dataset = load_iris()

x = dataset.data
y = dataset.target
# print(x.shape,y.shape) (150, 4) (150,)

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x,y,train_size = 0.8, shuffle = True, random_state = 1)
x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size = 0.8, shuffle = True, random_state = 1)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],2,2,1)
x_test = x_test.reshape(x_test.shape[0],2,2,1)
x_val = x_val.reshape(x_val.shape[0],2,2,1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Dropout, Flatten

input = Input(shape = (2,2,1))
d = Conv2D(100, 2, activation ='relu', padding = 'same')(input)
d = Dropout(0.3)(d)
d = Conv2D(100, 2)(d)
d = Dropout(0.2)(d)
d = Flatten()(d)
d = Dense(50)(d)
d = Dense(3, activation = 'softmax')(d)

model = Model(inputs = input, outputs = d)
model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 1000, validation_data = (x_val,y_val), callbacks = [es], batch_size = 1)

#4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

# loss :  0.13219477236270905
# acc :  0.9666666388511658