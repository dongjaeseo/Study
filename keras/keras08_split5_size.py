# practice >> validation_data
# train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,train_size = 0.7 , shuffle = False) # shuffle = True >> pick 60% random
#train_size = 0.9, test_size = 0.2
#train_size = 0.7, test_size = 0.2
# above two cases, comment the results of each

# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8, shuffle = True) # ~= test_size = 0.2 

print(x_train.shape)
print(x_train)
# print(x_val.shape)
print(x_test.shape)
print(x_test)

'''
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
model.fit(x_train,y_train,epochs= 100, validation_data = (x_val,y_val)) # validation splitted

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

# validation_data
# loss:  0.0007012678543105721
# mae:  0.02229570783674717

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict): # 여기서 y프레딕트는 x테스트의 결과값이므로 y테스트와 유사한 값이 나온다
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ", RMSE(y_test,y_predict))
# print("MSE: ", mean_squared_error(y_test,y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2: ",r2)

# RMSE:  0.003188248866737651
# R2:  0.9999999856619414
'''

# Case1: train_size + test_size = 1.1 >> Gives an error stating : ValueError: The sum of test_size and train_size = 1.1 << reduce test/train size
# Case2: train_size + test_size < 1 (Shuffle:False) >> First give values to train_size and consecutively to test_size >> leftovers from the end
# >> order doesnt matter , train size given priority