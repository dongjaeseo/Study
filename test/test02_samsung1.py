import numpy as np
import pandas as pd
 
def split_x(seq,size):
    a = []
    for i in range(len(seq)-size+1):
        subset = seq[i:i+size]
        a.append(subset)
    return np.array(a)  


#1. 데이터
df = pd.read_csv('./test/삼성전자.csv', index_col = None, header = 0, encoding = 'cp949')

df['시가'] = df.loc[:,['시가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['고가'] = df.loc[:,['고가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['저가'] = df.loc[:,['저가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['종가'] = df.loc[:,['종가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['거래량'] = df.loc[:,['거래량']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['금액(백만)'] = df.loc[:,['금액(백만)']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['개인'] = df.loc[:,['개인']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['기관'] = df.loc[:,['기관']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['외인(수량)'] = df.loc[:,['외인(수량)']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['외국계'] = df.loc[:,['외국계']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
df['프로그램'] = df.loc[:,['프로그램']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
# 스트링 형태의 수들을 실수로 변환

data_y = df.to_numpy()
data_y = data_y[:, [4]] 
df['target'] = data_y # 종가를 데이터프레임 끝에 추가해준다

data = df.to_numpy()
data_xy = data[:662, 1:] # 실제로 쓸 데이터만 추출

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(data_xy[:, :-1]) # 종가값 y 를 제외한 값들을 전처리해준다
data_xy[:, :-1] = scale.transform(data_xy[:, :-1])

df = pd.DataFrame(data_xy)
df.columns = ['start','high','low','close','fluctation','volume','amount','credit_cost','individual','agency','foreigner','foreign','program','foreigner_rate','target']

size = 5
df_x = df.iloc[1:, :-1]
df_y = df.iloc[:-(size), [-1]]
df_x_pred = df.iloc[0:size, :-1]

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
model.add(LSTM(512, activation = 'relu', input_shape = (size,14)))
model.add(Dropout(drop))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(1))
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience= 50)
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/samsungjuga_{epoch:3d}-{val_loss:.0f}.hdf5', monitor = 'val_loss', save_best_only=True)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 5000, validation_data = (x_val,y_val), batch_size = 8, callbacks = [es,cp])

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size=8)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
print("loss : ", result[0])
print("R2 : ", r2_score(y_pred,y_test))
y_next = model.predict(x_pred)
print("내일의 종가는??? : ", y_next)

# loss :  1133674.625
# R2 :  0.9868319053626069
# 내일의 종가는??? :  [[93828.39]]





