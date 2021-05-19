# load best model

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:6000].reshape(-1, 28, 28, 1)/255.
x_test = x_test[:1000].reshape(-1, 28, 28, 1)/255.
y_train = y_train[:6000]
y_test = y_test[:1000]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = load_model('C:/data/h5/autokeras/keras103.h5')
best_model = load_model('C:/data/h5/autokeras/keras105.h5')
best_model.summary()

results = model.evaluate(x_test, y_test)
print(results)

best_results = best_model.evaluate(x_test, y_test)
print(best_results)

print('====done====')