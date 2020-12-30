import numpy as np
# 1#다


#1. data
x = np.array([range(100)])
y = np.array([range(711,811), range(1,101), range(201,301)])
x = np.transpose(x)
y = np.transpose(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, shuffle = True, random_state =66)

import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

#1. 데이터
x = np.array([range(1,101)])
y = np.array([range(201,301),range(301,401)])
x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, shuffle = False)

#2. 모델링
model = Sequential()
model.add(Dense(5,input_dim = 1, activation= 'relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

# input = Input(shape=(1,))
#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 150, validation_split = 0.2, batch_size = 1)

#4. 평가 예측
loss,mae = model.evaluate(x_test,y_test,batch_size = 1)
print('loss : ', loss)
print('mae : ', mae)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import mean_squared_error
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_pred, y_test)
print('R2 : ',r2)