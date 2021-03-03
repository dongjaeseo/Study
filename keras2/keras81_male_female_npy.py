# fit
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential

#1. 데이터

x_train = np.load('../data/image/gender_generator/train/x.npy')
y_train = np.load('../data/image/gender_generator/train/y.npy')
x_val = np.load('../data/image/gender_generator/test/x.npy')
y_val = np.load('../data/image/gender_generator/test/y.npy')

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

#2. 모델
vgg16 = VGG16(include_top= False, input_shape = (256,256,3))
vgg16.trainable = False

model = Sequential()
# model.add(UpSampling2D(size = (2,2)))
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation= 'sigmoid'))


#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.25)
model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer = 'adam')
model.fit(x_train,y_train, validation_data = (x_val, y_val), epochs = 1000, callbacks = [es,lr])

#4. 평가 예측
from sklearn.metrics import accuracy_score
loss, mae = model.evaluate(x_val, y_val)
print('acc : ', mae)

# acc :  0.619596540927887