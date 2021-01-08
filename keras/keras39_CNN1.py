from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters = 10,kernel_size=(2,2), strides =2, padding = 'same', input_shape = (10,10,1)))
# model.add(MaxPooling2D(pool_size = (2,3))) # 풀사이즈로 나눠준다 각 행과 열을
model.add(Conv2D(9,(2,2), padding ='valid'))
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten()) # 차원을 바꿔줌, 평평하게 해준다 해서 플래튼, conv랑 덴스랑 엮어줌
model.add(Dense(1))

model.summary()