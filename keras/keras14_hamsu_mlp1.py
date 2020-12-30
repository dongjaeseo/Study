# 다#1
# keras10_mlp2.py 를 함수형으로 바꾸시오.

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터
x = np.array([range(1,101),range(101,201)])
y = np.array([range(201,301)])
x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, shuffle = False)

#2. 모델링
input = Input(shape=(2,))
den1 = Dense(5,activation = 'relu')(input)
den2 = Dense(5)(den1)
den3 = Dense(5)(den2)
den4 = Dense(5)(den3)
output = Dense(1)(den4)

model = Model(inputs = input, outputs = output)
model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 100, validation_split = 0.2, batch_size = 1)

#4. 평가 예측
loss,mae = model.evaluate(x_test,y_test,batch_size = 1)
print('loss : ', loss)
print('mae : ', mae)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_pred, y_test)
print('R2 : ',r2)
