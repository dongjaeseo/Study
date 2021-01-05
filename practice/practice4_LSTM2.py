import numpy as np

#1. data
x = np.array([1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8,9,8,9,10,9,10,11,10,11,12,20,30,40,30,40,50,40,50,60])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

x = x.reshape(13,3)
x_pred = x_pred.reshape(1,3)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x)
x = scale.transform(x)
x_pred = scale.transform(x_pred)
x = x.reshape(13,3,1)
x_pred = x_pred.reshape(1,3,1)

#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

input = Input(shape =(3,1))
d = LSTM(100, activation = 'relu')(input)
d = Dense(40)(d)
d = Dense(40)(d)
d = Dense(10)(d)
d = Dense(20)(d)
d = Dense(1)(d)

model = Model(inputs = input, outputs= d)
model.summary()

#3. compile fit
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor ='loss', patience = 15)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y,epochs = 3000, verbose = 2, batch_size =1, callbacks = early)

#4. evaluation prediction
loss = model.evaluate(x,y,batch_size = 1)
print('loss : ', loss)
y_pred = model.predict(x_pred)
print(y_pred)

# [[82.32007]]
# [[79.84583]]