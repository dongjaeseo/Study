import numpy as np

#1. data
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# scaler = QuantileTransformer() # 균등분포
scaler = QuantileTransformer(output_distribution='normal') # 정규분포
scaler.fit(x)
x = scaler.transform(x)

# print(np.max(x), np.min(x)) # 711.0 0.0 => 1.0 0.0
# print(np.max(x[0]))


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size =0.8, shuffle = True)

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
model.fit(x_train,y_train,epochs = 100, batch_size = 8, validation_split = 0.2, verbose = 2)

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

