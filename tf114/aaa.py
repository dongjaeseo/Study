import numpy as np

# 0 1 0 찾아주기
'''
y = [[0.8, 0.1, 0.1],
     [0.6, 0.2, 0.2],
     [0.3, 0.5, 0.2]]

print(np.where([a == np.max(a) for a in y], 1, 0))
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense
model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape = (8, 8, 3)))
model.add(Dense(10))
model.summary()