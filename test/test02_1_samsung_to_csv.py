import numpy as np
import pandas as pd

df = pd.read_csv('./test/삼성전자.csv', index_col = None, header = 0, encoding = 'cp949')
# print(df.shape) # (2400,15)
df.columns = ['date','start','high','low','close','fluctation','volume','amount','credit_cost','individual','agency','foreigner','foreign','program','foreigner_rate']
# 컬럼명 영어로 변경

df['start'] = df.loc[:,['start']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['high'] = df.loc[:,['high']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['low'] = df.loc[:,['low']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['close'] = df.loc[:,['close']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['volume'] = df.loc[:,['volume']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['amount'] = df.loc[:,['amount']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['individual'] = df.loc[:,['individual']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['agency'] = df.loc[:,['agency']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['foreigner'] = df.loc[:,['foreigner']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['foreign'] = df.loc[:,['foreign']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['program'] = df.loc[:,['program']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)


df1 = df.iloc[1:662 , 1:] # 시고저종 등등등
dft = df.iloc[:661 , [4]] # 종가 , 내일의 종가를 예측하기 위해 x 에 대응하는 y 는 다음날의 y 라서 데이터의 행이다름
dfx = df.iloc[[0], 1:] # 오늘의 시고저종 데이터


dft.columns = ['target']

df1 = df1.iloc[::-1]  # 데이터 거꾸로~
dft = dft.iloc[::-1]

# df1[0:5] = pd.to_numeric(df1[0:5], errors='coerce')
# df1.to_csv('./test/ss_price.csv', sep = ',')
# dft.to_csv('./test/ss_target.csv', sep = ',')


x = df1.to_numpy()
y = dft.to_numpy()
x_pred = dfx.to_numpy()
# print(x[0], y[0])
# print(x.shape) # (661, 14)
# print(y.shape) # (661, 1)

def split_x(seq,size):
    a = []
    for i in range(len(seq)-size+1):
        subset = seq[i:i+size]
        a.append(subset)
    return np.array(a)    
    
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size= 0.8,shuffle = True)
x_train,x_val,y_train,y_val = tts(x_train,y_train,train_size= 0.8,shuffle = True)

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
x_val = scale.transform(x_val)
x_pred = scale.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)

# np.save('../data/npy/ss_x_train.npy', arr = x_train)
# np.save('../data/npy/ss_x_test.npy', arr = x_test)
# np.save('../data/npy/ss_x_val.npy', arr = x_val)
# np.save('../data/npy/ss_x_pred.npy', arr = x_pred)

# np.save('../data/npy/ss_y_train.npy', arr = y_train)
# np.save('../data/npy/ss_y_test.npy', arr = y_test)
# np.save('../data/npy/ss_y_val.npy', arr = y_val)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
drop = 0.3
model = Sequential()
model.add(LSTM(128, activation = 'relu', input_shape = (14,1)))
model.add(Dropout(drop))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(1))

model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience= 20)
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/samsungjuga_{epoch:3d}.hdf5', monitor = 'val_loss', save_best_only=True)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 1000, validation_data = (x_val,y_val), batch_size = 8, callbacks = [es,cp])

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size=8)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
print("loss : ", result[0])
print("R2 : ", r2_score(y_pred,y_test))
y_tom = model.predict(x_pred)
print("내일의 종가는??? : ", y_tom)


# R2 :  0.9297658741274074
# 내일의 종가는??? :  [[97527.35]]

# R2 :  0.9544995812013763
# 내일의 종가는??? :  [[95312.69]]

# R2 :  0.9233492996307531
# 내일의 종가는??? :  [[96962.92]]

# R2 :  0.9205352228611048
# 내일의 종가는??? :  [[98964.625]]

# R2 :  0.7414319589257756
# 내일의 종가는??? :  [[89080.266]]

# loss :  2698262.75
# R2 :  0.9337886851060314
# 내일의 종가는??? :  [[90325.82]]

# loss :  7821113.5
# R2 :  0.8277320116109961
# 내일의 종가는??? :  [[81798.79]]

# loss :  1223667.625
# R2 :  0.9754239507198131
# 내일의 종가는??? :  [[95209.74]]