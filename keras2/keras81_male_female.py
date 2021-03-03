# 전이학습을 통해 이전에 한 남녀 구분모델을 만들어볼것이다!
# 실습 VGG16 으로 만들어봐!

# fit
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential

#1. 데이터
train_datagen = ImageDataGenerator(
    rotation_range = 5,
    height_shift_range= 0.2,
    width_shift_range=0.2,
    validation_split=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

xy_train = train_datagen.flow_from_directory(
    directory='../data/image/gender',
    class_mode = 'binary',
    batch_size = 8,
    target_size=(128,128),
    subset= 'training'
)

xy_val = train_datagen.flow_from_directory(
    directory='../data/image/gender',
    class_mode = 'binary',
    batch_size = 8,
    target_size=(128,128),
    subset= 'validation'
)

#2. 모델
vgg16 = VGG16(include_top = False, input_shape = (128,128,3))
vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.25)
model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer = 'adam')
model.fit(xy_train, steps_per_epoch= 174, validation_data= xy_val, validation_steps= 44, epochs = 1000, callbacks = [es,lr])

#4. 평가 예측
from sklearn.metrics import accuracy_score
loss, mae = model.evaluate(xy_val, steps = 2)
print('acc : ', mae)

# acc :  0.619596540927887