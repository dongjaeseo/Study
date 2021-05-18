# 오토케라스에서 이미지 분류해보자

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:6000].reshape(6000, 28, 28, 1)/255.
x_test = x_test[:1000].reshape(1000, 28, 28, 1)/255.
y_train = y_train[:6000]
y_test = y_test[:1000]

model = ak.ImageClassifier(
    overwrite = True,
    max_trials = 3,
    loss = 'mse',
    metrics = ['acc']
)

# model.summary()
# > summary가 안 먹힌다 >> 오토케라스는 모델을 만들겠다는 계획이고 아직 만들어진건 X

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 4)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 2, factor = 0.5, verbose = 1)
filepath = 'C:/data/h5/autokeras/'
mc = ModelCheckpoint(filepath, save_best_only=True, verbose = 1)

model.fit(x_train, y_train, epochs = 2, validation_split = 0.2, callbacks = [es, lr, mc])

results = model.evaluate(x_test, y_test)

print(results)