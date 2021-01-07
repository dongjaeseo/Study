import numpy as np

from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data     # (506,13)
y = datasets.target   # (506,)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,train_size = 0.8, shuffle = True)
x_train,x_val,y_train,y_val =train_test_split(x_train,y_train,train_size = 0.8, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)   # (323,13,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)       # (102,13,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)           # (81,13,1)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape = (13,1)))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 500, validation_data = (x_val,y_val), callbacks = [early], verbose = 2)

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
print("R2 : ", r2_score(y_pred,y_test))

# loss :  [18.665754318237305, 3.338855743408203]
# R2 :  0.8006743000971055