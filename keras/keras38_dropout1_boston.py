# EarlyStopping

import numpy as np

#1. data
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size =0.8, shuffle = True)

x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size = 0.8, shuffle = True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. modelling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
input = Input(shape = (13,))
d = Dense(64,activation = 'relu')(input)
d = Dropout(0.3)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(1)(d)
model = Model(inputs = input, outputs = d)

#3. compile fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # min, max, auto > confused > auto 

model.fit(x_train,y_train,epochs = 2000, batch_size = 8, validation_data = (x_val,y_val), verbose = 1, callbacks= [early_stopping])

#4. evaluation prediction
loss, mae = model.evaluate(x_test,y_test,batch_size = 8)
print('loss : ', loss)
print('mae : ', mae)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

print('RMSE : ', rmse(y_pred,y_test))
print('R2 : ', r2_score(y_pred,y_test))

# before preprocessing
# RMSE :  5.689833076488374
# R2 :  0.4123716592763268

# after preprocessing # x = x / 711.
# RMSE :  4.102010926070412
# R2 :  0.7431972895240151

# after preprocessing # x MinMaxScaler
# RMSE :  3.4024146070682932
# R2 :  0.8595388453921976

# after preprocessing # x_train MinMaxScaler (validation_split)
# RMSE :  2.8876899794402093
# R2 :  0.8758000393945378

# after preprocessing # x_train MinMaxScaler (validation_data)
# RMSE :  3.25432761341863
# R2 :  0.8986712422091289


# after tune
# RMSE :  2.2407017353681886
# R2 :  0.9443237249040412

# after dropout
# RMSE :  3.18260210582049
# R2 :  0.869462309774332