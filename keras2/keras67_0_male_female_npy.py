# '../data/image/gender/female, male' <<

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

# train_datagen = ImageDataGenerator(
#     rotation_range = 5,
#     height_shift_range= 0.1,
#     width_shift_range=0.1,
#     validation_split=0.2,
#     fill_mode='nearest',
#     rescale= 1/255.
# )

# test_datagen = ImageDataGenerator(
#     rescale= 1/255.
# )

# xy_train = train_datagen.flow_from_directory(
#     directory='../data/image/gender',
#     class_mode = 'binary',
#     batch_size = 5000,
#     target_size=(128,128),
#     subset= 'training'
# )

# xy_val = train_datagen.flow_from_directory(
#     directory='../data/image/gender',
#     class_mode = 'binary',
#     batch_size = 5000,
#     target_size=(128,128),
#     subset= 'validation'
# )

# np.save('../data/image/gender_generator/train/x.npy', arr = xy_train[0][0])
# np.save('../data/image/gender_generator/train/y.npy', arr = xy_train[0][1])
# np.save('../data/image/gender_generator/test/x.npy', arr = xy_val[0][0])
# np.save('../data/image/gender_generator/test/y.npy', arr = xy_val[0][1])

x_train = np.load('../data/image/gender_generator/train/x.npy')
y_train = np.load('../data/image/gender_generator/train/y.npy')
x_test = np.load('../data/image/gender_generator/test/x.npy')
y_test = np.load('../data/image/gender_generator/test/y.npy')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)