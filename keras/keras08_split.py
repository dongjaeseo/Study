from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터

x = np.array(range(1,101))
# x = np.array(range(100))
y = np.array(range(101,201))

x_train = x[:60] # 순서 0 부터 59 번째까지 :::: 값 1 ~ 60
x_val = x[60:80] # 순서 60 부터 79번째까지 :::: 값 61 ~ 80
x_test = x[80:]  # 값 81 ~ 100
# 리스트의 슬라이싱
y_train = y[:60] # 순서 0 부터 59 번째까지 :::: 
y_val = y[60:80] # 순서 60 부터 79번째까지 :::: 
y_test = y[80:]  # 값 81 ~ 100



#2. 모델링
model = Sequential()
model.add(Dense(10,input_dim = 1,activation='relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train,epochs =100, batch_size = 1, validation_data=(x_val,y_val))

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

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2: ",r2)
# accuracy 랑은 엄연히 다른 지표지만 1에 근접할수록 좋다
