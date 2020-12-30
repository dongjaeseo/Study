from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터

x = np.array(range(1,101))
# x = np.array(range(100))
y = np.array(range(1,101))

# x_train = x[:60] # 순서 0 부터 59 번째까지 :::: 값 1 ~ 60
# x_val = x[60:80] # 순서 60 부터 79번째까지 :::: 값 61 ~ 80
# x_test = x[80:]  # 값 81 ~ 100
# 리스트의 슬라이싱
# y_train = y[:60] # 순서 0 부터 59 번째까지 :::: 
# y_val = y[60:80] # 순서 60 부터 79번째까지 :::: 
# y_test = y[80:]  # 값 81 ~ 100



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, shuffle = True) # shuffle = True >> pick 60% random ,shuffle = False >> pick 60percent from beginning
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. modelling
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(1))

#3. compile, fit
model.compile(loss='mse',optimizer='adam', metrics = 'mae')
model.fit(x_train,y_train,epochs= 100, validation_split = 0.2) # validation splitted

#4. evaluation, prediction
loss, mae = model.evaluate(x_test,y_test)
print('loss: ', loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
print(y_predict)

# shuffle = False
# loss:  5.3550098527921364e-05
# mae:  0.00723953265696764

# shuffle = True
# loss:  1.0329372344131116e-05
# mae:  0.002687859581783414