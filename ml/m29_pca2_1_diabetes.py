import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(x.shape,y.shape) (442, 10) (442,)

pca = PCA(n_components=8)
x2 = pca.fit_transform(x)
print(x2)
# print(x2.shape) (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) 

# 7개 0.9479436357350414
# 8개 0.9913119559917797
## 압축률!

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)

d = np.argmax(cumsum >= 0.95)+1
print('cumsum >= 0.95', cumsum>=0.95)
print('d : ', d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()