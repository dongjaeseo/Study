# fit
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

#1. 데이터

x_train = np.load('../data/image/gender_generator/train/x.npy')
y_train = np.load('../data/image/gender_generator/train/y.npy')
x_test = np.load('../data/image/gender_generator/test/x.npy')
y_test = np.load('../data/image/gender_generator/test/y.npy')

#2. 모델
model = Sequential()
model.add(Conv2D(128, 3, padding = 'same', activation= 'relu', input_shape = (128,128,3)))
model.add(Conv2D(64, 5, padding = 'same', activation= 'relu'))
model.add(Conv2D(64, 5, padding = 'same', activation= 'relu'))
model.add(MaxPooling2D(3))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.25)
model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer = 'adam')
model.fit(x_train,y_train, validation_split=0.2, epochs = 1000, callbacks = [es,lr])

#4. 평가 예측
from sklearn.metrics import accuracy_score
loss, mae = model.evaluate(x_test,y_test)
print('acc : ', mae)

# acc :  0.619596540927887