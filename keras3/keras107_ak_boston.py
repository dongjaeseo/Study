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