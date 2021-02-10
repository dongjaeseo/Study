# 실습 
# cifar10을 flow 로 구성해서 완성
# ImageDataGenerator / fit_generator 를 쓸것

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split

#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8)

train_datagen = ImageDataGenerator(rescale = 1/255., width_shift_range=0.2, height_shift_range=0.2)
test_datagen = ImageDataGenerator(rescale = 1/255.)

batch = 25
train_xy = train_datagen.flow(x_train,y_train,batch_size = batch)
val_xy = test_datagen.flow(x_val,y_val,batch_size = batch)
test_xy = test_datagen.flow(x_test,y_test,batch_size = batch)

#2. 모델
model = Sequential()
model.add(Conv2D(128, 3, padding = 'same', activation= 'relu', input_shape = (32,32,3)))
model.add(Conv2D(64, 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(64, 5, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(3))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(patience = 10)
lr = ReduceLROnPlateau(factor = 0.25, patience= 5)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['acc'])
hist = model.fit_generator(train_xy, validation_data= val_xy, epochs = 1000, steps_per_epoch = 40000/batch,
         validation_steps=10000/batch, callbacks = [es,lr])
        
#4. 평가
print('accuracy : ', model.evaluate(x_test, y_test)[1])

# accuracy :  0.42980000376701355