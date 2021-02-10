# '../data/image/gender/female, male' <<

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential

train_datagen = ImageDataGenerator(
    height_shift_range= 0.2,
    width_shift_range=0.2,
    validation_split=0.2,
    fill_mode='nearest',
    rescale= 1/255.
)

test_datagen = ImageDataGenerator(
    rescale= 1/255.
)

batch =40
xy_train = train_datagen.flow_from_directory(
    directory='../data/image/gender',
    class_mode = 'binary',
    batch_size = batch,
    target_size=(128,128),
    subset= 'training'
)

xy_val = train_datagen.flow_from_directory(
    directory='../data/image/gender',
    class_mode = 'binary',
    batch_size = batch,
    target_size=(128,128),
    subset= 'validation'
)

#2. 모델
model = Sequential()
model.add(Conv2D(128, 3, padding = 'same', activation= 'relu', input_shape = (128,128,3)))
model.add(Conv2D(64, 5, padding = 'same', activation= 'relu'))
model.add(MaxPooling2D(3))
model.add(Conv2D(128, 3, padding = 'same', activation= 'relu'))
model.add(Conv2D(64, 3, padding = 'same', activation= 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.25)
cp = ModelCheckpoint('../data/modelcheckpoint/gender_classification.hdf5', save_best_only=True)
history = model.fit_generator(xy_train, steps_per_epoch=xy_train.samples/batch, epochs = 1000, verbose = 1, validation_data= xy_val, validation_steps=xy_val.samples/batch, callbacks = [es, lr, cp])

#4. 평가
print('val_acc : ', history.history['val_acc'][-1])

