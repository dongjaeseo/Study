import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_x(seq,size):
    a = []
    for i in range(len(seq)-size+1):
        subset = seq[i:i+size]
        a.append(subset)
    return np.array(a)  

# 새로운 데이터
df = pd.read_csv("./samsung/삼성전자2.csv", index_col = None, header = 0, encoding = 'cp949', thousands = ',')
df.drop(['일자','전일비','Unnamed: 6','등락률','신용비','외인(수량)','외국계','프로그램','개인','기관','외인비'], axis = 1, inplace = True)
# print(df.columns) # Index(['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '외인비']
df.dropna(axis = 0, inplace = True)
df.drop(1, axis = 0, inplace = True)

# 기존 데이터
df_old = pd.read_csv('./samsung/삼성전자.csv',index_col = None, header = 0, encoding = 'cp949',thousands=',')
df_old.drop(['일자','등락률','신용비','외인(수량)','외국계','프로그램','개인','기관','외인비'], axis = 1, inplace = True)
df_old.dropna(axis = 0, inplace = True)

# 데이터 병합
df_new = df.append(df_old, ignore_index = True)

# 액면분할
df_new.iloc[663:, :4] /= 50.
df_new.iloc[663:,[4]] *= 50.
df_new['target'] = df_new.iloc[:,[3]]
df_new = df_new.astype(float)
dataset = df_new.to_numpy()

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(dataset[:, :-1]) # 종가값 y 를 제외한 값들을 전처리해준다
dataset[:, :-1] = scale.transform(dataset[:, :-1])

df = pd.DataFrame(dataset)

size = 15
df_x = df.iloc[1:, :-1] # 상관계수 높은 7개만 사용
df_y = df.iloc[:-(size), [-1]]
df_x_pred = df.iloc[0:size+8, :-1]

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

np.save('./samsung/samsung_new.npy', arr = [x_train,x_test,x_val,y_train,y_test,y_val,x_pred])
#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
model = Sequential()
drop = 0.25
model.add(Conv1D(256, 2, activation = 'relu', padding ='same', input_shape = (size,x_train.shape[2])))
model.add(Dropout(drop))
model.add(Conv1D(128,2,activation = 'relu', padding ='same'))
model.add(Dropout(drop))
model.add(Conv1D(64,2,activation = 'relu', padding ='same'))
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(drop))
model.add(Dense(256,activation = 'relu'))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1))
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience= 40)
cp = ModelCheckpoint(filepath = './samsung/samsungjuga_0115.hdf5', monitor = 'val_loss', save_best_only=True)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit(x_train,y_train,epochs = 1000, validation_data = (x_val,y_val), batch_size = 8, callbacks = [es,cp])

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 8)
y_next = model.predict(x_pred)
print("내일의 주가는? ", y_next)

plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()
