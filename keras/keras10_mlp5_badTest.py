# practice
# 1. R2 : 0 < < 0.5
# 2. layer: >=5
# 3. node: >=10
# 4. batch_size <= 8
# 5. epochs >=30
import numpy as np

#1. data
x = np.array([range(1,101), range(301,401), range(401,501), range(701,801), range(201,301)])
y = np.array([range(1201,1301), range(1401,1501)])
x = np.transpose(x)
y = np.transpose(y)
x_pred2 = np.array([200,500,600,900,400])
x_pred2 = x_pred2.reshape(1,5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, shuffle = False)


#2. modelling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_dim = 5))
model.add(Dense(500))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(2))

#3. compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 30, validation_split = 0.2, batch_size =8)

#4. evaluation, prediction
loss, mae = model.evaluate(x_test,y_test, batch_size = 8)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print(y_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict): # 여기서 y프레딕트는 x테스트의 결과값이므로 y테스트와 유사한 값이 나온다
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE: ", RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2: ",r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)

# RMSE:  5.100207165701493
# R2:  0.21768080802788414
# [[1402.777 1618.347]]