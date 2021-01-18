import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
model = Sequential()
model.add(Dense(1000,input_dim = 1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer = Adam(lr = 0.1)
# loss :  0.002560411114245653 결과물 :  [[11.065999]] // 0.1
# loss :  6.856737129034424e-14 결과물 :  [[10.999998]] // 0.01
# loss :  2.222317334599211e-07 결과물 :  [[10.999063]] // 0.001
# optimizer = Adam(lr = 0.0001)
# epoch 가 부족하다!

# optimizer = Adadelta(lr = 0.001)
# loss :  0.08957868814468384 결과물 :  [[10.476841]] // 0.1
# loss :  0.0003318395174574107 결과물 :  [[10.965798]] // 0.01
# loss :  11.185042381286621 결과물 :  [[5.0085545]] // 0.001

# optimizer = Adamax(lr = 0.001)
# loss :  34.755149841308594 결과물 :  [[1.7789114]] // 0.1
# loss :  1.0483347696876866e-11 결과물 :  [[10.999996]] // 0.01
# loss :  2.174260798530961e-13 결과물 :  [[11.000002]] // 0.001

# optimizer = Adagrad(lr = 0.001)
# loss :  3.528758796278453e-08 결과물 :  [[10.999891]] // 0.1
# loss :  3.3703094004522427e-07 결과물 :  [[11.001245]] // 0.01
# loss :  5.7427540014032274e-05 결과물 :  [[10.987675]] // 0.001

# optimizer = RMSprop(lr = 0.001)
# loss :  217.90969848632812 결과물 :  [[10.510212]] // 0.1
# loss :  8.26978874206543 결과물 :  [[13.486481]] // 0.01
# loss :  0.0003854793612845242 결과물 :  [[11.032956]] // 0.001

# optimizer = SGD(lr = 0.0001)
# loss :  nan 결과물 :  [[nan]] // ????? >> 튕겨나오는거임 0.1 , 0.01
# loss :  1.605083866706991e-06 결과물 :  [[10.999016]] // 0.001
# loss :  0.001873303554020822 결과물 :  [[10.936315]] // 0.0001

# optimizer = Nadam(lr = 0.001)
# loss :  2069010.375 결과물 :  [[-2478.3252]] // 0.1
# loss :  2.444267119434962e-13 결과물 :  [[10.999999]] // 0.01
# loss :  4.697460642688611e-09 결과물 :  [[11.000103]] // 0.001

#### 모델에 맞는 optimizer 를 찾는다, lr 은 낮을수록 좋으나 너무 낮을 경우 epoch 이 따라오지 못하는 경우가 있다

model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])
model.fit(x,y,epochs = 100, batch_size = 1)

#4. 평가 예측
loss,mse = model.evaluate(x,y,batch_size=1)
y_pred = model.predict([11])
print('loss : ', loss, '결과물 : ', y_pred)