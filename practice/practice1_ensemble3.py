import numpy as np

#1. 데이터
x1 = np.array([range(1,101),range(101,201),range(201,301),range(301,401)])
x2 = np.array([range(401,501),range(501,601),range(601,701),range(701,801)])
y1 = np.array([range(51,151),range(151,251)])
y2 = np.array([range(251,351),range(351,451)])
y3 = np.array([range(451,551),range(551,651)])

xtest1 = np.array([[0,100,200,300]])
xtest2 = np.array([[400,500,600,700]])
ytest1 = np.array([[50,150]])
ytest2 = np.array([[250,350]])
ytest3 = np.array([[450,550]])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test,y3_train,y3_test =\
     train_test_split(x1,x2,y1,y2,y3,shuffle = False, train_size = 0.8)

#2. 모델링
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input1 = Input(shape=(4,))
d1 = Dense(10,activation = 'relu')(input1)
d1 = Dense(10)(d1)

input2 = Input(shape=(4,))
d2 = Dense(10,activation = 'relu')(input2)
d2 = Dense(9)(d2)
d2 = Dense(4)(d2)

from tensorflow.keras.layers import concatenate
merge = concatenate([d1,d2])
d3 = Dense(10)(merge)
d3 = Dense(5)(d3)

d4 = Dense(5)(d3)
d4 = Dense(9)(d4)
d4 = Dense(2)(d4)

d5 = Dense(6)(d3)
d5 = Dense(10)(d5)
d5 = Dense(10)(d5)
d5 = Dense(2)(d5)

d6 = Dense(5)(d3)
d6 = Dense(5)(d6)
d6 = Dense(2)(d6)

model = Model(inputs = [input1,input2], outputs = [d4,d5,d6])
# model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit([x1_train,x2_train],[y1_train,y2_train,y3_train],batch_size = 1, epochs =100, validation_split =0.2)

#4. 평가 예측
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test,y3_test],batch_size = 1)
print('loss : ', loss)

y_pred = model.predict([x1_test,x2_test])

from sklearn.metrics import mean_squared_error
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse1 = rmse(y_pred[0],y1_test)
rmse2 = rmse(y_pred[1],y2_test)
rmse3 = rmse(y_pred[2],y3_test)

rmse = (rmse1+rmse2+rmse3)/3
print('RMSE : ', rmse)

from sklearn.metrics import r2_score
r21 = r2_score(y_pred[0],y1_test)
r22 = r2_score(y_pred[1],y2_test)
r23 = r2_score(y_pred[2],y3_test)
r2 = (r21+r22+r23)/3
print('R2 : ', r2)

ypred = model.predict([xtest1,xtest2])
print(ypred[0])
print(ypred[1])
print(ypred[2])

# RMSE :  0.22652648672394204
# R2 :  0.9983041688224977
# [[ 50.050785 150.71832 ]]
# [[251.45175 350.3697 ]]
# [[450.0555 550.3556]]