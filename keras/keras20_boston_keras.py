# make 2 files
# 1. best fit model without earlystopping
# 2. best fit model with earlystopping 

from tensorflow.keras.datasets import boston_housing

import numpy as np

(x_train,y_train),(x_test,y_test) = boston_housing.load_data()

#1. data

from sklearn.model_selection import train_test_split as tts
x_train,x_val,y_train,y_val = tts(x_train,y_train, train_size = 0.8, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input = Input(shape = (13,))
d = Dense(64, activation = 'relu')(input)
d = Dense(100, activation = 'relu')(d)
d = Dense(100, activation = 'relu')(d)
d = Dense(100, activation = 'relu')(d)
d = Dense(100, activation = 'relu')(d)
d = Dense(64, activation = 'relu')(d)
d = Dense(1)(d)

model = Model(inputs = input, outputs = d)

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auto')

#3. compile fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 500, validation_data = (x_val,y_val), batch_size = 8)

#4. evaluation prediction
loss, mae = model.evaluate(x_test,y_test,batch_size = 8)
print('loss : ', loss)
print('mae : ', mae)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
print("R2 : ", r2_score(y_pred,y_test))


# mae :  2.2727205753326416
# R2 :  0.7845398337371695