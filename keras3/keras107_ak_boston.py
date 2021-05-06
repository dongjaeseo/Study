import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist, boston_housing
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# (404, 13) (404,)
# (102, 13) (102,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
'''
model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=3
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(patience = 6)
lr = ReduceLROnPlateau(patience = 3, factor = 0.5, verbose = 1)
filepath = 'C:/data/h5/autokeras/'
mc = ModelCheckpoint(filepath, save_best_only=True, verbose = 1)

model.fit(x_train, y_train, epochs = 10, validation_split=0.2, callbacks = [es, lr, mc])

results = model.evaluate(x_test, y_test)

print(results)

best_model = model.tuner.get_best_model()
best_model.save('C:/data/h5/autokeras/keras107.h5')

model2 = model.export_model()
try:
    model2.save('C:/data/h5/autokeras/keras107', save_format='tf')
except:
    model2.save('C:/data/h5/autokeras/keras107.h5')

'''

model3 = load_model('C:/data/h5/autokeras/keras107', custom_objects=ak.CUSTOM_OBJECTS)
result_boston = model3.evaluate(x_test, y_test)

y_pred = model3.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('load_result :', result_boston, r2)

print('끝끄르ㅡ틑끝')