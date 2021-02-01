import numpy as np
from tensorflow.keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x = np.append(x_train, x_test, axis = 0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

########################################################################
pca = PCA(n_components= len(x[0]))
x2 = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) +1
print(d)

pca = PCA(n_components= d)
x2 = pca.fit_transform(x)

x_train = x2[:60000]
x_test = x2[60000:]
########################################################################

parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[4,5,6]},
    {'n_jobs' : [-1]}
]
print('haha')
model = RandomizedSearchCV(XGBClassifier(), parameters, cv = 5)
print('haha')
model.fit(x_train,y_train, eval_metric = 'merror', verbose = True, eval_set = [(x_train,y_train), (x_test, y_test)])
print('haha')

print('모델 스코어는? : ', model.score(x_test, y_test))