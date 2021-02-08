import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale = 1./255)

## 이건 이런식으로 바꿔준다 선언만 한거고

# flow 또는 flow_from_directory 가 실질적인 변환

# train_generator 
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),
    batch_size = 200,
    class_mode='binary'
)                           # (80, 150, 150, 1)
# Found 160 images belonging to 2 classes.

# test_generator 
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150),
    batch_size = 200,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.

model = Sequential()
model.add(Conv2D(128, 3, padding='same', activation='relu', input_shape = (150,150,3)))
model.add(Conv2D(64, 5, padding='same', activation='relu'))
model.add(Conv2D(64, 5, padding='same', activation='relu'))
model.add(MaxPooling2D(3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.25, patience = 5)
history = model.fit_generator(
    xy_train, steps_per_epoch=32, epochs = 100, validation_data=xy_test, validation_steps=4, callbacks = [es,lr]
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화!!

plt.plot(acc)
plt.plot(val_acc)
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['train_acc','val_acc','train_loss','val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss & acc')
plt.show()