import numpy as np
# 다#1

#1. data
x = np.array([range(100),range(301,401),range(1,101)])
y = np.array(range(711,811))
x = np.transpose(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, shuffle = True, random_state =66)

# print(x.shape)  #(10,) >> (2,10)
# print(x)
# print(y.shape)

#2. modelling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense :: 가능하지만 조금 느리다
model = Sequential()
model.add(Dense(10, input_dim = 3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.summary()


#3. compile fit
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs =100, batch_size =1, validation_split = 0.2)

#4. evaluation prediction
loss,mae = model.evaluate(x_test,y_test)
print('loss : ',loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
# print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict): # 여기서 y프레딕트는 x테스트의 결과값이므로 y테스트와 유사한 값이 나온다
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ", RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2: ",r2)

