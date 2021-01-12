import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_pred = x_test[:10]

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8, shuffle = True)

# print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# print(x_train[0])
# print(y_train[0])
# print(x_train[0].shape) # (28,28)

# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0]) # gray 안넣어주면 컬러로 나오는데 제대로 된건 아님
# plt.show()

# 민맥스 스케일을 못 쓰므로 /255. 해준다
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],x_pred.shape[2],1)/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)/255.
# (x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


#2. 모델링
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 80, kernel_size = (2,2), padding = 'same', strides = 1, input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64,2))
model.add(Dropout(0.2))
model.add(Conv2D(64,2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
# model.summary()

# model.save('../data/h5/k52_1_model1.h5')

#3. 컴파일 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# k52_1_mnist_??? => k52_1_MCK.h5 이름을 바꿔줄것
# es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
# cp = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss', save_best_only=True, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
# hist = model.fit(x_train,y_train, epochs = 100, batch_size = 32 ,validation_data=(x_val,y_val), verbose = 2, callbacks = [es,cp])

# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

# model1 = load_model('../data/h5/k52_1_model2.h5')

# #4-1. 평가 예측
# loss = model1.evaluate(x_test,y_test,batch_size = 32)
# print('model1_loss : ', loss[0])
# print('model1_acc : ', loss[1])

#4-2. 평가 예측
model.load_weights('../data/h5/k52_1_weight.h5')

loss = model.evaluate(x_test,y_test,batch_size = 32)
print('가중치_loss : ', loss[0])
print('가중치_acc : ', loss[1])

# 가중치_loss :  0.09655730426311493
# 가중치_acc :  0.97079998254776

model2 = load_model('../data/h5/k52_1_model2.h5')

loss2 = model2.evaluate(x_test,y_test,batch_size = 32)
print('로드모델_loss : ', loss2[0])
print('로드모델_acc : ', loss2[1])

# 로드모델_loss :  0.09655730426311493
# 로드모델_acc :  0.97079998254776