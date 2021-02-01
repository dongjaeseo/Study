import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

pca = PCA(n_components=30)
x2 = pca.fit_transform(x)
print(x2)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) 

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)

d = np.argmax(cumsum >= 0.95)+1
print('cumsum >= 0.95', cumsum>=0.95)
print('d : ', d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()