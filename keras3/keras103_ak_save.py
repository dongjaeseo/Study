import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:6000].reshape(-1, 28, 28, 1)/255.
x_test = x_test[:1000].reshape(-1, 28, 28, 1)/255.
y_train = y_train[:6000]
y_test = y_test[:1000]

model = ak.ImageClassifier(
    overwrite = True,
    max_trials = 1,
    loss = 'mse',
    metrics = ['acc']
)

es = EarlyStopping(patience = 4)
lr = ReduceLROnPlateau(factor = 0.25, patience = 2, verbose = 1)
filepath = 'C:/data/h5/autokeras/'
mc = ModelCheckpoint(filepath, save_best_only=True, verbose = 1)

model.fit(x_train, y_train, epochs = 1, validation_split = 0.2, callbacks = [es, lr, mc])

results = model.evaluate(x_test, y_test)

print(results)

# 모델 save 를 해볼건데
# AttributeError: 'ImageClassifier' object has no attribute 'save'

# >> 이를 모델 형태로 바꾸는 녀석은 expot_model
model2 = model.export_model()
model2.save('C:/data/h5/autokeras/keras103.h5')

model2.summary()