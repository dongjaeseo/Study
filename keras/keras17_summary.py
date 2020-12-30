import numpy as np
import tensorflow as tf

#1. data 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. modelling 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation = 'linear'))
model.add(Dense(5, activation = 'linear'))
model.add(Dense(8))
model.add(Dense(1))

model.summary()

# 실습2 + 과제
# ensemble 1,2,3,4에 대해 서머리를 계산하고
# 이해한것을 과제로 제출할것
# layer를 만들때 'name' 에 대해 확인하고 설명 << layer에 네임을 지정할 수 있는데 붙이는 이유를 찾아서 설명