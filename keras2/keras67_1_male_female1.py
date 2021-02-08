# fit generator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    width_shift_range= 0.2,
    height_shift_range= 0.2,
)
test_datagen = ImageDataGenerator(
    validation_split = 0.2
)

train_batch = 1389
test_batch = 347
iteration = 3

xy_train = test_datagen.flow_from_directory(
    directory='../data/image/gender',
    target_size = (128,128),
    class_mode = 'binary',
    batch_size = train_batch,
    shuffle = True,
    seed = 42,
    subset = 'training',
    save_to_dir='../data/image/gender_generator/train/'
)

xy_test = test_datagen.flow_from_directory(
    directory='../data/image/gender',
    target_size = (128,128),
    class_mode = 'binary',
    batch_size = test_batch,
    shuffle = True,
    seed = 42,
    subset = 'validation',
    save_to_dir='../data/image/gender_generator/test/'
)

for i, (img,label) in xy_train:
    if i is iteration-1:
        break
