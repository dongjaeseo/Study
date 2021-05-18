# 오토케라스에서 모델 로드하기

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:6000].reshape(-1, 28, 28, 1)/255.
x_test = x_test[:1000].reshape(-1, 28, 28, 1)/255.
y_train = y_train[:6000]
y_test = y_test[:1000]

from tensorflow.keras.models import load_model
model = load_model('C:/data/h5/autokeras/keras103.h5')
model.summary()
