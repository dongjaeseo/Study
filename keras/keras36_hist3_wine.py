import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
x = dataset.data
y = dataset.target

# print(x.shape) # (178, 13)
# print(y.shape) # (178,)

# 실습, DNN 완성할것!!

from sklearn.model_selection import train_test_split as tts
x,x_test,y,y_test = tts(x,y,train_size = 0.8, shuffle = True)
x_train,x_val,y_train,y_val = tts(x,y,train_size = 0.8, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

input = Input(shape= (13,))
d = Dense(100, activation = 'relu')(input)
d = Dense(10)(d)
d = Dense(100)(d)
d = Dense(100)(d)
d = Dense(100)(d)
d = Dense(50)(d)
d = Dense(3, activation = 'softmax')(d)
model = Model(inputs = input, outputs = d)

model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience= 15, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train,y_train,epochs = 1500, validation_data = (x_val,y_val), batch_size = 1,callbacks = es)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

'''
#4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size = 1)
print('loss : ', loss)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis = 1)
y_test = np.argmax(y_test,axis = 1)

print(y_pred)
print(y_test)

from sklearn.metrics import accuracy_score
print('acc : ', accuracy_score(y_test,y_pred))

# acc :  1.0
# acc :  0.9722222222222222
'''