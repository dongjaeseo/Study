# 과제 및 실습  Dense
# 전처리, EarlyStopping 등등 다 넣을것!
# 데이터 1 ~ 100 / 5개씩 자르고 
#     x             y
# 1,2,3,4,5         6
# .....
# 95,96,97,98,99   100

# predict 를 만들것
# 96,97,98,99,100 -> 101
# ...
# 100,101,102,103,104 -> 105
# (101,102,103,104,105)

import numpy as np

# 함수 가져오기


def split_x(seq,size):
    a = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        a.append(subset)
    return np.array(a)

dataset = np.array(range(1,101))
pred_data = np.array(range(96,106))
size = 6
datasets = split_x(dataset, size)
pred_data = split_x(pred_data,size)

x = datasets[: , :-1]
y = datasets[: , -1:]
x_pred = pred_data[: , :-1]
# print(x.shape,y.shape) # (95,5)  (95,1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle = True, train_size =0.8)


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_pred = scale.transform(x_pred)
x_test = scale.transform(x_test)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
input = Input(shape = (5,))
d = Dense(30, activation = 'relu')(input)
d = Dense(10)(d)
d = Dense(50)(d)
d = Dense(20)(d)
d = Dense(1)(d)

model = Model(inputs = input, outputs = d)
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'loss', patience=15, mode = 'auto')

model.compile(loss ='mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs = 1000, validation_split = 0.2, batch_size = 1, callbacks =[early])

#4. 평가 예측
print('loss : ', model.evaluate(x_test,y_test,batch_size =1))
y_pred = model.predict(x_pred)

print(y_pred)

# LSTM
# loss :  [0.05852947011590004, 0.19898465275764465]
# [[100.945984]
#  [102.03147 ]
#  [103.12516 ]
#  [104.22713 ]
#  [105.337425]]

# DNN  성능적으로도 DNN이좋게나왔다,,, 속도는 확실히 DNN
# loss :  [0.005130778066813946, 0.029096001759171486]
# [[101.02738 ]
#  [102.02767 ]
#  [103.027985]
#  [104.02827 ]
#  [105.02857 ]]