# m31로 만든 0.95 이상의 n_component = ? 를 사용하여
# dnn 모델을 만들것

# mnist dnn 보다 성능 좋게!!!
# cnn 과 비교!!

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

(x_train,y_train),(x_test,y_test) = mnist.load_data()

# scale = MinMaxScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

x = np.append(x_train,x_test, axis = 0)
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])

pca = PCA(n_components = len(x[0]))
x2 = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95)+1
print('d : ', d)

pca = PCA(n_components = d)
x2 = pca.fit_transform(x)

x_train = x[:60000]
x_test = x[60000:]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation= 'relu', input_shape = (x_train.shape[1],)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.25)
es = EarlyStopping(monitor = 'val_loss', patience = 10)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 32, epochs = 1000, callbacks = [es], validation_split = 0.2)

#4. 평가 예측
model.evaluate(x_test,y_test,batch_size= 32)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)

from sklearn.metrics import accuracy_score
print('acc score : ', accuracy_score(y_test,y_pred))

# acc score :  0.9684