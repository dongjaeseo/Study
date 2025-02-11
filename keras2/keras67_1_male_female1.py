# '../data/image/gender/female, male' <<

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential

train_datagen = ImageDataGenerator(
    rotation_range = 5,
    height_shift_range= 0.2,
    width_shift_range=0.2,
    validation_split=0.2,
    fill_mode='nearest',
    rescale= 1/255.
)

test_datagen = ImageDataGenerator(
    rescale= 1/255.
)

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

# print(xy_train[0][0].shape) (1389, 128, 128, 3)
# print(xy_train[0][1].shape)(1389,)
# print(xy_val[0][0].shape)(347, 128, 128, 3)
# print(xy_val[0][1].shape)(347,)

#2. 모델
model = Sequential()
model.add(Conv2D(128, 3, padding = 'same', activation= 'relu', input_shape = (128,128,3)))
model.add(Conv2D(64, 5, padding = 'same', activation= 'relu'))
model.add(Conv2D(64, 5, padding = 'same', activation= 'relu'))
model.add(MaxPooling2D(3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.25)
history = model.fit_generator(xy_train, steps_per_epoch=len(xy_train[0][0])/8, epochs = 1000, verbose = 1, validation_data= xy_val, validation_steps=len(xy_val[0][0])/8, callbacks = [es, lr])

#4. 평가
print('acc : ', history.history['val_acc'][-1])