import numpy as np

#1. 데이터
x1 = np.array([range(1,101),range(101,201),range(201,301)])
x2 = np.array([range(301,401),range(401,501),range(501,601)])
y1 = np.array([range(601,701),range(701,801),range(801,901)])
y2 = np.array([range(901,1001),range(1001,1101),range(1101,1201)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test = train_test_split(x1,x2,y1,y2, train_size = 0.8, shuffle = False)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 모델1
input1 = Input(shape = (3,))
dense1 = Dense(10,activation = 'relu')(input1)
dense1 = Dense(10)(dense1)
dense1 = Dense(10)(dense1)

# 모델2
input2 = Input(shape = (3,))
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(10)(dense2)

# 병합
from tensorflow.keras.layers import concatenate
merge = concatenate([dense1, dense2])
middle = Dense(10)(merge)
middle = Dense(10)(middle)

# 분기1
dense3 = Dense(10)(middle)
dense3 = Dense(10)(dense3)
dense3 = Dense(3)(dense3)

# 분기2
dense4 = Dense(10)(middle)
dense4 = Dense(3)(dense4)

model = Model(inputs = [input1,input2], outputs = [dense3,dense4])
model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs = 100, batch_size =1, validation_split = 0.2)

#4. 평가 예측
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test],batch_size = 1)
print("loss : ", loss)

y_pred = model.predict([x1_test,x2_test])

from sklearn.metrics import mean_squared_error
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

rmse1 = rmse(y_pred[0],y1_test)
rmse2 = rmse(y_pred[1],y2_test)
print('RMSE : ', (rmse1+rmse2)/2)

from sklearn.metrics import r2_score

r2_1 = r2_score(y_pred[0],y1_test)
r2_2 = r2_score(y_pred[1],y2_test)
print('R2 : ', (r2_1+r2_2)/2)

print(y_pred[1])
print(y2_test)

# RMSE :  0.13871049154932802
# R2 :  0.9993961079502897
