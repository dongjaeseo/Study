# 행렬 데이터를 슬라이싱하기
# DNN 으로 만들기

import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(seq,size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(a,size)

x = dataset[: , :-1]
y = dataset[: , -1:]
# print(x.shape) # (6,4)
# print(y.shape) # (6,1)

x_pred = np.array([[7,8,9,10]])


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x)
scale.transform(x)
scale.transform(x_pred)

# x = x.reshape(6,4,1)
# x_pred = x_pred.reshape(1,4,1)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (4,)))
model.add(Dense(20))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x,y,epochs = 400, verbose = 2, batch_size = 1)

#4. 평가 예측
loss = model.evaluate(x,y,batch_size =1)
print('loss : ', loss)

y_pred = model.predict(x_pred)
print(y_pred)

# loss :  [0.00012694811448454857, 0.010285377502441406]
# [[10.967938]]
# loss :  [0.0001894552115118131, 0.01330113410949707]
# [[11.013206]]

# DNN
# loss :  [8.860907655616757e-06, 0.0023682911414653063]
# [[11.025495]]
# loss :  [0.304023802280426, 0.5441588759422302]
# [[11.712528]]











