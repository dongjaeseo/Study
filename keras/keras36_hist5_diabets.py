# 3. MinMaxScaler on whole x

#1. data
import numpy as np
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (442, 10) (442,)

x = x - np.min(x)

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size = 0.8,shuffle = True)
x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size = 0.8, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
haha = MinMaxScaler()
haha.fit(x_train)
x_train = haha.transform(x_train)
x_test = haha.transform(x_test)
x_val = haha.transform(x_val)

#2. modelling
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
input = Input(shape = (10,))
d = Dense(64,activation = 'relu')(input)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(1)(d)
model = Model(inputs = input, outputs = d)

#3. compile fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=15, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit(x_train,y_train,epochs = 1500, batch_size = 16, validation_split = 0.2, verbose = 2, callbacks = [es])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])

plt.title('diabetes')
plt.xlabel('epochs')
plt.ylabel('loss, mae')
plt.legend(['train loss','val loss','train mae', 'val mae'])
plt.show()

'''
#4. evaluation prediction
loss, mae = model.evaluate(x_test,y_test,batch_size = 16)
print('loss : ', loss)
print('mae : ', mae)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))
print("RMSE : ", rmse(y_pred,y_test))
print("R2 : ", r2_score(y_pred,y_test))
'''


#1. before preprocessing
# RMSE :  63.2536780496614
# R2 :  0.08630641532455097

#2. after setting range of x between 0 and 1
# RMSE :  56.35428236167508
# R2 :  0.11204699174472899

#3. after using MinMaxScaler
# RMSE :  54.15666617070038
# R2 :  0.2306576620370565
