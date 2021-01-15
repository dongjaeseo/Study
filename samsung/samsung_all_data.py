import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_x(seq,size):
    a = []
    for i in range(len(seq)-size+1):
        subset = seq[i:i+size]
        a.append(subset)
    return np.array(a)  


#1. 데이터
df = pd.read_csv('./samsung/삼성전자.csv', index_col = None, header = 0, encoding = 'cp949', thousands=',')

data_y = df.to_numpy()
data_y = data_y[:, [4]] 
df['target'] = data_y # 종가를 데이터프레임 끝에 추가해준다
df.loc[662:,['시가','고가','저가','종가','target']] = df.loc[662:,['시가','고가','저가','종가','target']]/50.
df.loc[662:,['거래량']] = df.loc[662:,['거래량']]*50.
df = df.dropna(axis = 0, how = 'any') # 결측치가 있는 행 전체 삭제

data = df.to_numpy()
data_xy = data[:, 1:] # 실제로 쓸 데이터만 추출


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(data_xy[:, :-1]) # 종가값 y 를 제외한 값들을 전처리해준다
data_xy[:, :-1] = scale.transform(data_xy[:, :-1])

df = pd.DataFrame(data_xy)
df.columns = ['start','high','low','close','fluctation','volume','amount','credit_cost','individual','agency','foreigner','foreign','program','foreigner_rate','target']

# 상관계수
# start	0.99
# high	0.99
# low	0.99
# close	0.99
# amount	0.69
# volume	0.5
# foreigner_rate	0.21
# fluctation	0.16
# individual	0.13

size = 12
df_x = df.iloc[1:, [0,1,2,3,5,6]] # 상관계수 높은 6개만 사용 , individual 사용
df_y = df.iloc[:-(size), [-1]]
df_x_pred = df.iloc[0:size+8, [0,1,2,3,5,6]]

df_x = df_x.iloc[::-1]
df_y = df_y.iloc[::-1]
df_x_pred = df_x_pred.iloc[::-1] # 데이터 순서를 거꾸로 해준다

x = df_x.to_numpy().astype(float)
y = df_y.to_numpy().astype(float)
x_pred = df_x_pred.to_numpy().astype(float)

x = split_x(x,size)
x_pred = split_x(x_pred,size)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, shuffle = True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8, shuffle = True)

print(x_train.shape,x_test.shape,x_val.shape,y_train.shape,y_test.shape,y_val.shape,x_pred.shape)

np.savez('./samsung/samsung.npz', x_train = x_train, x_test = x_test, x_val = x_val,y_train = y_train,y_test = y_test, y_val = y_val, x_pred = x_pred)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
drop = 0.3
model = Sequential()
model.add(Conv1D(512, 2, activation = 'relu', padding ='same', input_shape = (size,x_train.shape[2])))
model.add(MaxPooling1D(2))
model.add(Dropout(drop))
model.add(Conv1D(256,2,activation = 'relu', padding ='same'))
model.add(MaxPooling1D(2))
model.add(Dropout(drop))
model.add(Conv1D(128,2,activation = 'relu', padding ='same'))
model.add(Dropout(drop))
model.add(Flatten())
model.add(Dense(4096,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(1024,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(32,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(8,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(1))
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience= 50)
cp = ModelCheckpoint(filepath = './samsung/samsungjuga.hdf5', monitor = 'val_loss', save_best_only=True)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit(x_train,y_train,epochs = 1000, validation_data = (x_val,y_val), batch_size = 32, callbacks = [es,cp])

#4. 평가 예측
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()
# result = model.evaluate(x_test,y_test,batch_size=32)
# y_pred = model.predict(x_test)

# from sklearn.metrics import r2_score
# print("loss : ", result[0])
# print("R2 : ", r2_score(y_pred,y_test))
# y_next = model.predict(x_pred)
# print("내일의 종가는??? : ", y_next)

# loss :  1133674.625
# R2 :  0.9868319053626069
# 내일의 종가는??? :  [[93828.39]]

