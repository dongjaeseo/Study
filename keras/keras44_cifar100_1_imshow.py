import numpy as np
from tensorflow.keras.datasets import cifar100
(x_train,y_train),(x_test,y_test) = cifar100.load_data()


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()