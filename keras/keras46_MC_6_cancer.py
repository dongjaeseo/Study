import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size = 0.8, shuffle = True)
x_test,x_pred,y_test,y_actual = tts(x_test,y_test,train_size = 0.8, shuffle = True)

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
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(100,activation = 'relu', input_dim = 30))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.summary()

#3. compile fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
cp = ModelCheckpoint(filepath = './ModelCheckPoint/k46_6_cancer_{epoch:3d}-{val_acc:.3f}.hdf5', monitor = 'val_acc', save_best_only=True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
earlystopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto')
hist = model.fit(x_train,y_train,epochs = 1000, verbose = 2, validation_split = 0.2, callbacks = [earlystopping, cp])

#4. evaluation prediction
loss = model.evaluate(x_test,y_test)
print('accuracy : ', loss[1])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('비용 & 정확도')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

y_pred = model.predict(x_pred)
y_act = np.argmax(y_pred, axis = 1)