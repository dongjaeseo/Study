import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape)(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)(10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train))255 0
# print(np.min(y_train),np.max(y_train)) 0,9

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8, shuffle = True, random_state = 0)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

model = Sequential()
model.add(Conv2D(64, 3, padding = 'same', activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))
model.add(Conv2D(64,3))
model.add(Conv2D(64,3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()

#3. 컴파일
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './ModelCheckPoint/k46_1_fashion_{epoch:03d}-{val_acc:.3f}.hdf5'

cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_acc', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor = 'val_loss', patience = 20)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train,y_train,epochs = 1000,batch_size = 64, validation_data = (x_val,y_val), callbacks = [es,cp])

#4. 평가 예측
loss = model.evaluate(x_test,y_test,batch_size = 64)
print('acc : ', loss[1])

# acc :  0.904699981212616
