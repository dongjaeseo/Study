# 오토케라스

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)/255.
x_test = x_test.reshape(-1, 28, 28, 1)/255.

# imageclassifier 에서는 원핫해도 되고 안해도 된다

model = ak.ImageClassifier(
    overwrite = True,
    max_trials = 1
)

model.fit(x_train, y_train, epochs = 1)

results = model.evaluate(x_test, y_test)

print(results)