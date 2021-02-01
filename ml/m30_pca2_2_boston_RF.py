import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model = RandomForestRegressor()
model.fit(x_train,y_train)

print('PCA 전! : ', model.score(x_test,y_test))

#############################################################################################

# pca = PCA(n_components=len(x[0]))
# x2 = pca.fit_transform(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.99)+1

import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca = PCA(n_components=11)
x2 = pca.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x2,y,train_size = 0.8)

model = RandomForestRegressor()
model.fit(x_train,y_train)

print('PCA 후! : ', model.score(x_test,y_test))

# PCA 전! :  0.34003973450492564
# PCA 후! :  0.5259310236996946
