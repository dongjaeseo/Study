import numpy as np

#1. data
x1 = np.array([range(1,101),range(101,201)])
x2 = np.array([range(201,301),range(301,401)])
y = np.array([range(401,501)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)
x1test = np.array([[101,201]])
x2test = np.array([[301,401]])

from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(x1,x2,y,shuffle = False, train_size = 0.8)

#2. 모델링

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input1 = Input(shape = (2,))
d1 = Dense(15,activation = 'relu')(input1)
d1 = Dense(10)(d1)

input2 = Input(shape = (2,))
d2 = Dense(10,activation = 'relu')(input2)
d2 = Dense(15)(d2)
d2 = Dense(7)(d2)

from tensorflow.keras.layers import concatenate
merge = concatenate([d1,d2])
d3 = Dense(10)(merge)
d3 = Dense(15)(d3)
d3 = Dense(10)(d3)
d3 = Dense(1)(d3)

model = Model(inputs = [input1,input2], outputs = d3)
# model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics =['mae'])
model.fit([x1_train,x2_train],y_train,epochs = 100, batch_size = 1, validation_split = 0.2)

#4. 평가 예측
loss = model.evaluate([x1_test,x2_test],y_test,batch_size = 1)
y_pred = model.predict([x1_test,x2_test])

from sklearn.metrics import mean_squared_error
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse = rmse(y_pred,y_test)
print('RMSE : ', rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_pred,y_test)
print('R2 : ', r2)

ypred = model.predict([x1test,x2test]) # 501에 근접하게 나와야함
print(ypred)

# RMSE :  0.006550834681703034
# R2 :  0.9999987064353996
# [[500.9883]]
