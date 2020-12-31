# practice : 19_1 ~ 19_5, EarlyStopping
# make 6 files
# tune first one and print the results of five

# 1. original data

#1. data
import numpy as np
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (442, 10) (442,)

print(np.max(x),np.min(x)) # 0.198787989657293 -0.137767225690012

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size = 0.8,shuffle = True)

#2. modelling
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape = (10,)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))
model.summary()

#3. compile fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 100, batch_size = 8, validation_split = 0.2)

#4. evaluation prediction
loss, mae = model.evaluate(x_test,y_test,batch_size = 8)
print('loss : ', loss)
print('mae : ', mae)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))
print("RMSE : ", rmse(y_pred,y_test))
print("R2 : ", r2_score(y_pred,y_test))

#1. before preprocessing
# RMSE :  63.2536780496614
# R2 :  0.08630641532455097