# '../data/image/gender/female, male' <<

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import matplotlib.pyplot as plt

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

batch = 8
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

img = Image.open(
    '../data/image/IU.jpg'
)
pix = np.asarray(img)
pix = np.resize(pix,(128,128,3))
pix = pix.reshape(1,128,128,3)
x_pred = test_datagen.flow(
    pix, batch_size = batch
)

# print(xy_train[0][0].shape) (1389, 128, 128, 3)
# print(xy_train[0][1].shape)(1389,)
# print(xy_val[0][0].shape)(347, 128, 128, 3)
# print(xy_val[0][1].shape)(347,)

#2. 모델
model = load_model('../data/modelcheckpoint/gender_classification.hdf5')
# model = Sequential()
# model.add(Conv2D(128, 3, padding = 'same', activation= 'relu', input_shape = (128,128,3)))
# model.add(Conv2D(64, 5, padding = 'same', activation= 'relu'))
# model.add(Conv2D(64, 5, padding = 'same', activation= 'relu'))
# model.add(MaxPooling2D(3))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation= 'relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer = 'adam')
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience = 20)
# lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.25)
# cp = ModelCheckpoint('../data/modelcheckpoint/gender_classification.hdf5', save_best_only=True)
# history = model.fit_generator(xy_train, steps_per_epoch=len(xy_train[0][0])/batch, epochs = 1000, verbose = 1, validation_data= xy_val, validation_steps=len(xy_val[0][0])/batch, callbacks = [es, lr, cp])

#4. 평가

pred = model.predict_generator(x_pred)
result = np.where(pred<0.5, '여자!', '남자!')
print(result)
if pred>0.5:
    print(f'남자일 확률 {pred*100}%!')
else:
    pred = 1- pred
    print(f'여자일 확률 {pred*100}%!')