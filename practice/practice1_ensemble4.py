import numpy as np

#1. 데이터
x = np.array([range(1,101)])
y1 = np.array([range(101,201)])
y2 = np.array([range(201,301)])

x = np.transpose(x)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x_train,x_test,y1_train,y1_test,y2_train,y2_test = train_test_split(x,y1,y2,train_size = 0.8, shuffle = True)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
input = Input(shape = (1,))
d1 = Dense(10, activation = 'relu')(input)
d1 = Dense(5)(d1)
d1 = Dense(5)(d1)
d1 = Dense(5)(d1)

d2 = Dense(5)(d1)
d2 = Dense(5)(d2)
d2 = Dense(1)(d2)

d3 = Dense(5)(d1)
d3 = Dense(5)(d3)
d3 = Dense(1)(d3)

model = Model(inputs = input, outputs = [d2,d3])
# model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,[y1_train,y2_train],epochs = 100, batch_size = 1)

#4. 평가 예측
loss = model.evaluate(x_test,[y1_test,y2_test])
print('loss : ', loss)

y_pred = model.predict(x_test)
from sklearn.metrics import mean_squared_error
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse1 = rmse(y_pred[0],y1_test)
rmse2 = rmse(y_pred[1],y2_test)
rmse = (rmse1+rmse2)/2
print("RMSE : ", rmse)

from sklearn.metrics import r2_score

r21 = r2_score(y_pred[0],y1_test)
r22 = r2_score(y_pred[1],y2_test)
r2 = (r21+r22)/2
print("R2 : ", r2)

# loss :  [0.0020011705346405506, 0.0006593956495635211, 0.0013417748268693686, 0.02047424390912056, 0.030257415026426315]
# RMSE :  0.031154472550123886
# R2 :  0.9999981744140068