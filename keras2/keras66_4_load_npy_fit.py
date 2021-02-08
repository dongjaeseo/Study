import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 실습!!
# 모델을 만들어라!!

