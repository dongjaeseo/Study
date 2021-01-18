import numpy as np
import pandas as pd

# 15일까지 병합된 데이터를 불러오자!
df1 = pd.read_csv('./samsung/삼성전자_병합.csv', encoding = 'cp949', header = 0, index_col = None, thousands = ',')

# 코스닥 선물인버스!
df2 = pd.read_csv('./samsung/KODEX 코스닥150 선물인버스.csv', encoding = 'cp949', header = 0, index_col = None, thousands = ',')
df2.drop(['일자','전일비','Unnamed: 6','등락률','개인','기관','외인(수량)','외국계','프로그램','외인비'],axis = 1,inplace = True)

# 데이터프레임에 먼저 시가를 붙여준다!
df2['target'] = df1.loc[:,'시가']
df_kodex = df2.iloc[2:,:-1]
y1 = df2.iloc[1:-1,-1].to_numpy()
y2 = df2.iloc[:-2,-1].to_numpy()
df_kodex['target1'] = y1
df_kodex['target2'] = y2
# print(df_kodex.shape) # (1086, 9)


# 상관계수 확인해보자
# print(df_kodex.corr()) # 시 고 저 종 거래량 금액 신용비 7개 피처

df1['target'] = df1.loc[:,'시가']
df_samsung = df1.iloc[2:,:-1]
y1 = df1.iloc[1:-1,-1].to_numpy()
y2 = df1.iloc[:-2,-1].to_numpy()
df_samsung['target1'] = y1
df_samsung['target2'] = y2
df_samsung = df_samsung.iloc[:df_kodex.shape[0]]

data_samsung = df_samsung.to_numpy()
data_kodex = df_kodex.to_numpy()

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(data_samsung[:,:-2])
data_samsung[:,:-2] = scale.transform(data_samsung[:,:-2])

scale.fit(data_kodex[:,:-2])
data_kodex[:,:-2] = scale.transform(data_kodex[:,:-2])

# 시퀀셜 데이터로 만들어주기1
size = 10

def split_x(seq,size):
    a = []
    for i in range(len(seq)-size+1):
        subset = seq[i:i+size]
        a.append(subset)
    return np.array(a)

# 시퀀셜 데이터로 만들어주기2
x_samsung_before = data_samsung[:,:-2]
# 사이즈에 맞춰서 내일의 시가 y1, 모레의 시가 y2 를 정해준다
y1 = data_samsung[:(data_samsung.shape[0]-size+1),-2]
y2 = data_samsung[:(data_samsung.shape[0]-size+1),-1]
x_samsung = split_x(x_samsung_before,size)
x_samsung_pred = split_x(data_samsung[:size,:-2],size)

x_kodex_before = data_kodex[:,:-2]
x_kodex = split_x(x_kodex_before,size)
x_kodex_pred = split_x(data_kodex[:size,:-2],size)

# 트레인 테스트 스플릿
from sklearn.model_selection import train_test_split as tts
x_samsung_train,x_samsung_test,y1_train,y1_test,y2_train,y2_test,x_kodex_train,x_kodex_test = tts(x_samsung,y1,y2,x_kodex, train_size = 0.8, shuffle = True)
x_samsung_train,x_samsung_val,y1_train,y1_val,y2_train,y2_val,x_kodex_train,x_kodex_val = tts(x_samsung_train,y1_train,y2_train,x_kodex_train, train_size = 0.8, shuffle = True)

np.save('./samsung/samsung_final_samsung.npy', arr = [x_samsung_train,x_samsung_test,x_samsung_val,y1_train,y1_test,y1_val,y2_train,y2_test,y2_val,x_samsung_pred])
np.save('./samsung/samsung_final_kodex.npy', arr = [x_kodex_train,x_kodex_test,x_kodex_val,y1_train,y1_test,y1_val,y2_train,y2_test,y2_val,x_kodex_pred])

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense, MaxPooling1D, LSTM, Input, concatenate
drop = 0.3

# 삼성모델
input1 = Input(shape = (size,x_samsung_train.shape[2]))
d1 = Conv1D(512, 2, activation = 'relu', padding = 'same')(input1)
d1 = Dropout(drop)(d1)
d1 = Conv1D(256, 2, activation = 'relu', padding = 'same')(d1)
d1 = Conv1D(128, 2, activation = 'relu', padding = 'same')(d1)
d1 = Flatten()(d1)
d1 = Dense(512, activation = 'relu')(d1)
d1 = Dropout(drop)(d1)
d1 = Dense(256, activation = 'relu')(d1)
d1 = Dense(128, activation = 'relu')(d1)
d1 = Dense(32, activation = 'relu')(d1)

# 코덱스 모델
input2 = Input(shape = (size,x_kodex_train.shape[2]))
d2 = Conv1D(512, 2, activation = 'relu', padding = 'same')(input2)
d2 = Dropout(drop)(d2)
d2 = Conv1D(256, 2, activation = 'relu', padding = 'same')(d2)
d2 = Conv1D(128, 2, activation = 'relu', padding = 'same')(d2)
d2 = Flatten()(d2)
d2 = Dense(512, activation = 'relu')(d2)
d2 = Dropout(drop)(d2)
d2 = Dense(256, activation = 'relu')(d2)
d2 = Dense(128, activation = 'relu')(d2)
d2 = Dense(32, activation = 'relu')(d2)

# 병합
d3 = concatenate([d1,d2])
d3 = Dense(32, activation = 'relu')(d3)
d2 = Dropout(drop)(d3)
d3 = Dense(16, activation = 'relu')(d3)
d3 = Dense(8, activation = 'relu')(d3)
d3 = Dense(2)(d3)

model = Model(inputs = [input1,input2], outputs = d3)
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = './samsung/samsung_final.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True)
es = EarlyStopping(monitor = 'val_loss', patience = 30)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 15, factor = 0.5, verbose = 1)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit([x_samsung_train,x_kodex_train],[y1_train,y2_train],epochs = 1000,\
     batch_size = 4, validation_data = ([x_samsung_val,x_kodex_val],[y1_val,y2_val]),callbacks = [es,cp,lr])

#4. 평가 예측
result = model.evaluate([x_samsung_test,x_kodex_test],[y1_test,y2_test],batch_size = 4)
print('loss, mae = ', result)
y_pred = model.predict([x_samsung_pred,x_kodex_pred])
print(y_pred)

# 시각화
# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.legend('loss','val_loss')

# plt.show()
