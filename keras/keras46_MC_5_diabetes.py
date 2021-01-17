
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

x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size = 0.8, shuffle = True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. modelling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
input = Input(shape = (13,))
d = Dense(64,activation = 'relu')(input)
d = Dropout(0.2)(d)
d = Dense(64)(d)
d = Dropout(0.2)(d)
d = Dense(64)(d)
d = Dropout(0.2)(d)
d = Dense(64)(d)
d = Dropout(0.2)(d)
d = Dense(64)(d)
d = Dropout(0.2)(d)
d = Dense(64)(d)
d = Dense(1)(d)
model = Model(inputs = input, outputs = d)

#3. compile fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/k46_5_diabetes_{epoch:3d}-{val_loss:.3f}.hdf5',save_best_only=True, monitor = 'val_loss')
es = EarlyStopping(monitor = 'loss', patience=10, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 2000, batch_size = 8, validation_data = (x_val,y_val), verbose = 2, callbacks=[es,cp])

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

# R2 :  0.8076423445905687