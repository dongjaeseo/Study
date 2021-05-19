# binary classification using autokeras

import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = ak.StructuredDataClassifier(loss = 'binary_crossentropy', metrics = ['accuracy'], max_trials=2, overwrite=True)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(patience = 20)
lr = ReduceLROnPlateau(factor = 0.5, patience=10)

model.fit(x_train, y_train, epochs = 300, validation_split=0.2, callbacks=[es, lr])

results = model.evaluate(x_test, y_test)

model2 = model.export_model()
try:
    model2.save('ak_save_cancer', save_format = 'tf')
except:
    model2.save('ak_save_cancer.h5')

best_model = model.tuner.get_best_model()
try:
    best_model.save('ak_save_best_cancer', save_format = 'tf')
except:
    best_model.save('ak_save_best_cancer.h5')
    
from tensorflow.keras.models import load_model
model3 = load_model('ak_save_cancer', custom_objects=ak.CUSTOM_OBJECTS)
result_cancer = model3.evaluate(x_test, y_test)

model4 = load_model('ak_save_best_cancer', custom_objects=ak.CUSTOM_OBJECTS)
result_best_cancer = model4.evaluate(x_test, y_test)

print('result :', results)
print('load_result :', result_cancer)
print('load_best :', result_best_cancer)

# result : [0.3789757192134857, 0.9561403393745422]
# load_result : [0.3789757192134857, 0.9561403393745422]
# load_best : [0.3789757192134857, 0.9561403393745422]