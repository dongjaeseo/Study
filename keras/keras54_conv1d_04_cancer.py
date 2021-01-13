import numpy as np

x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')

# print(x.shape, y.shape) # (569, 30) (569,)

from sklearn.model_selection import train_test_split as tts
x,x_test,y,y_test = tts(x,y,train_size = 0.8, shuffle = True)
x_train,x_val,y_train,y_val = tts(x,y,train_size = 0.8, shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Input

input = Input(shape = (x_train.shape[1],1))
drop = 0.2
d = Conv1D(128,3,padding = 'valid', activation = 'relu')(input)
d = MaxPool1D(2)(d)
d = Dropout(drop)(d)
d = Conv1D(128,3,padding = 'valid', activation = 'relu')(d)
d = MaxPool1D(2)(d)
d = Dropout(drop)(d)
d = Flatten()(d)
d = Dense(64, activation = 'relu')(d)
d = Dropout(drop)(d)
d = Dense(64, activation = 'relu')(d)
d = Dropout(drop)(d)
d = Dense(64, activation = 'relu')(d)
d = Dropout(drop)(d)
d = Dense(1, activation = 'sigmoid')(d)

model = Model(inputs = input, outputs = d)
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 15)
# cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/k54_2_boston_{epoch:2d}-{val_loss:.3f}.hdf5', save_best_only=True)

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train,y_train,epochs = 1000, validation_data = (x_val,y_val), callbacks = [es], batch_size = 1)

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 1)
print('loss : ', result[0])
print('acc : ', result[1])

# from sklearn.metrics import r2_score
# y_pred = model.predict(x_test)
# print('r2 : ', r2_score(y_test,y_pred))

# loss :  0.12849050760269165
# acc :  0.9649122953414917

# loss :  0.12421248853206635
# acc :  0.9649122953414917