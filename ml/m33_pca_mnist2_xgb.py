import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x = np.append(x_train,x_test,axis = 0)
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])

pca = PCA(n_components= len(x[0]))
temp_x = pca.fit_transform(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 1) +1

pca = PCA(n_components= d)
x2 = pca.fit_transform(x)

x_train = x2[:60000]
x_test = x2[60000:]

scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

model = XGBClassifier(max_depth = 4, use_label_encoder= False)
model.fit(x_train,y_train, verbose = True)

print('모델 스코어!! : ', model.score(x_test,y_test))
print(f'784중 남은 피쳐수는 {d}!!!')

# 모델 스코어!! :  0.9558   
# 784중 남은 피쳐수는 713!!!