import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis = 0)
x = x.reshape(70000,784)

pca = PCA(n_components=784)
x2 = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)

d = np.argmax(cumsum >= 0.95)+1
print('cumsum >= 0.95', cumsum>=0.95)
print('d : ', d)

import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca = PCA(n_components=d)
x2 = pca.fit_transform(x)

x_train = x2[:60000]
x_test = x2[60000:]

(_, y_train), (_, y_test) = mnist.load_data()

model = RandomForestClassifier()

model.fit(x_train,y_train)
print(model.score(x_test,y_test))

# 0.9506