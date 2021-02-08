import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip= True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale = 1./255)

## 이건 이런식으로 바꿔준다 선언만 한거고

# flow 또는 flow_from_directory 가 실질적인 변환

# train_generator 
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),
    batch_size = 5,
    class_mode='binary'
)                           # (80, 150, 150, 1)
# Found 160 images belonging to 2 classes.

# test_generator 
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150),
    batch_size = 5,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000023697148550>
print(xy_train[0])
print(xy_train[0][0].shape) # (5, 150, 150, 3)
print(xy_train[0][1]) # [1. 1. 0. 1. 1.]
print(xy_train[0][1]) # [1. 1. 0. 1. 1.]
print(xy_train[0][1]) # [1. 1. 0. 1. 1.]
print(xy_train[0][1].shape) # (5,)

## 배치를 160으로 하면!!
# (160, 150, 150, 3)
# [1. 0. 0. 0. 1. 1. 0. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 1. 0.
#  1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 0. 1. 1. 0. 1.
#  1. 0. 0. 0. 0. 0. 0. 1. 0. 1. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 0. 0.
#  1. 0. 1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 0.
#  0. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0.
#  0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1.
#  1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 1. 0.]
# (160,)

## 배치를 260으로 하면!! 동일하다
# (160, 150, 150, 3)
# [1. 0. 0. 0. 1. 1. 0. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 1. 0.
#  1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 0. 1. 1. 0. 1.
#  1. 0. 0. 0. 0. 0. 0. 1. 0. 1. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 0. 0.
#  1. 0. 1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 0.
#  0. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0.
#  0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1.
#  1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 1. 0.]
# (160,)