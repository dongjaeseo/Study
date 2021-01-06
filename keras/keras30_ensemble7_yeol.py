# 행이 다른 앙상블 모델에 대해 공부!

import numpy as np

x1 = np.array([[1,2], [2,3], [3,4], [4,5], [5,6],
            [6,7], [7,8], [8,9], [9,10], [10,11],[20,30],[30,40],[40,50]]) # (13,2)
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],\
    [80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]]) # (13,3)

y1 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],\
    [80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]]) # (13,3)
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13,)

x1_pred = np.array([[55,65]])
x2_pred = np.array([[65,75,85]])

x1 = x1.reshape(x1.shape[0],x1.shape[1],1) # (13,2,1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1) # (13,3,1)
x1_pred = x1_pred.reshape(x1_pred.shape[0],x1_pred.shape[1],1) # (1,2,1)
x2_pred = x2_pred.reshape(x2_pred.shape[0],x2_pred.shape[1],1) # (1,3,1)

#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

# 모델1
input1 = Input(shape = (2,1))
d1 = LSTM(10, activation = 'relu')(input1)
d1 = Dense(10)(d1)
d1 = Dense(10)(d1)
d1 = Dense(5)(d1)

# 모델2
input2 = Input(shape = (3,1))
d2 = LSTM(12, activation = 'relu')(input2)
d2 = Dense(11)(d2)
d2 = Dense(11)(d2)
d2 = Dense(11)(d2)
d2 = Dense(7)(d2)

# 모델 병합
from tensorflow.keras.layers import concatenate
merge = concatenate([d1,d2])
d3 = Dense(80)(merge)
d3 = Dense(6)(d3)

# 분기1
d4 = Dense(30)(d3)
d4 = Dense(3)(d4)

# 분기2
d5 = Dense(40)(d3)
d5 = Dense(3)(d5)
d5 = Dense(1)(d5)

model = Model(inputs = [input1, input2], outputs = [d4,d5])
model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit([x1,x2],[y1,y2],epochs = 100, verbose = 2, batch_size =1)

#4. 평가 예측
print('loss : ', model.evaluate([x1,x2],[y1,y2],batch_size =1))
y_pred = model.predict([x1_pred,x2_pred])
print('y1_pred : ', y_pred[0])
print('y2_pred : ', y_pred[1])

# loss :  [0.012477551586925983, 0.08046843111515045]
# [[81.027145]]
# loss :  [0.37891313433647156, 0.41709110140800476]
# [[85.74817]]
# loss :  [0.3884856104850769, 0.35858017206192017]
# [[87.592445]]
