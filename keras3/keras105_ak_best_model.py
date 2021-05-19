# find best model 

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:6000].reshape(-1, 28, 28, 1)/255.
x_test = x_test[:1000].reshape(-1, 28, 28, 1)/255.
y_train = y_train[:6000]
y_test = y_test[:1000]

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,
    loss = 'mse',
    metrics = ['acc']
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 4)
lr = ReduceLROnPlateau(patience = 2, factor = 0.5, verbose = 1)
filepath = 'C:/data/h5/autokeras/'
mc = ModelCheckpoint(filepath, save_best_only=True, verbose =1)

model.fit(x_train, y_train, epochs = 1, validation_split=0.2, callbacks=[es,lr,mc])

results = model.evaluate(x_test, y_test)

print(results)

model2 = model.export_model()
model2.save('C:/data/h5/autokeras/keras103.h5')

best_model = model.tuner.get_best_model()
best_model.save('C:/data/h5/autokeras/keras105.h5')

print('=========Done=========')