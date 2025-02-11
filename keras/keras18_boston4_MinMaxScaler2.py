import numpy as np

#1. data
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape) # (506,13)
# print(y.shape) # (506,)
# print("=========================================")
# print(x[:5])
# print(y[:10])

# print(np.max(x), np.min(x)) # 711.0 0.0
# print(dataset.feature_names)
# print(dataset.DESCR)

# data preprocessing(minmax)
# x = x / 711. # divides all components in list of x by 711 where max of x[0], x[1] is not 711
# print(np.max(x[0]))

from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# print(np.max(x), np.min(x)) # 711.0 0.0 => 1.0 0.0  
# print(np.max(x[0]))

# minmaxscaler를 x 에 사용하면 0~1사이가 되는데 이러면 x_train이 0~1사이가 되는것이 아니기에
# x_train을 0~1 사이로 고정하고 그 스케일러에 다른 테스트, 프레딕션값을 트랜스폼해준다

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size =0.8, shuffle = True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. modelling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
input = Input(shape = (13,))
d = Dense(64,activation = 'relu')(input)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(64)(d)
d = Dense(1)(d)
model = Model(inputs = input, outputs = d)

#3. compile fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 100, batch_size = 16, validation_split = 0.2, verbose = 2)

#4. evaluation prediction
loss, mae = model.evaluate(x_test,y_test,batch_size = 8)
print('loss : ', loss)
print('mae : ', mae)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

print('RMSE : ', rmse(y_pred,y_test))
print('R2 : ', r2_score(y_pred,y_test))

# before preprocessing
# RMSE :  5.689833076488374
# R2 :  0.4123716592763268

# after preprocessing # x = x / 711.
# RMSE :  4.102010926070412
# R2 :  0.7431972895240151

# after preprocessing # x MinMaxScaler
# RMSE :  3.4024146070682932
# R2 :  0.8595388453921976

# after preprocessing # x_train MinMaxScaler
# RMSE :  2.8876899794402093
# R2 :  0.8758000393945378