from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x_train = array([1,2,3,4,5,6,7,8,9,10])
y_train = array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = array([16,17,18])


#2. 모델링
model = Sequential()
model.add(Dense(10,input_dim = 1,activation='relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train,epochs =100, batch_size = 1, validation_split=0.2)

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 1)
print("mse, mae: ", result)

y_predict = model.predict(x_test)
print("y_pred: ", y_predict)

# 사이킷 런 >> 머신러닝 라이브러리
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict): # 여기서 y프레딕트는 x테스트의 결과값이므로 y테스트와 유사한 값이 나온다
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ", RMSE(y_test,y_predict))
print("MSE: ", mean_squared_error(y_test,y_predict))

# 실습 RMSE를 0.1 이하로 낮추시오!