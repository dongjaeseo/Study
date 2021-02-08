import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import accuracy_score

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# def build_model(drop=0.5, optimizers = 'adam', act = 'relu', nodes = 64, layer = 2):
#     inputs = Input(shape = (28,28,1), name = 'input')
#     x = Conv2D(128, 3, activation = act, padding = 'same', name = 'conv1')(inputs)
#     x = Dropout(drop)(x)
#     for i in range(layer):
#         x = Conv2D(nodes, 3, activation = act, padding = 'same', name = f'conv2{i}')(x)
#     x = MaxPooling2D(3)(x)
#     x = Conv2D(nodes, 5, activation = act, padding = 'same', name = 'conv3')(x)
#     x = MaxPooling2D(5)(x)
#     x = Flatten()(x)
#     x = Dense(nodes, activation = act, name = 'hidden2')(x)
#     x = Dropout(drop)(x)
#     x = Dense(128, activation = act, name = 'hidden3')(x)
#     x = Dropout(drop)(x)
#     outputs = Dense(10, activation='softmax', name = 'outputs')(x)
#     model = Model(inputs = inputs, outputs = outputs)
#     model.compile(optimizers, metrics = ['acc'], loss = 'categorical_crossentropy')

#     return model

# def create_hyperparameter():
#     batches = [32, 64, 128]
#     optimizers = ['adam', 'rmsprop']
#     dropout = [0.2, 0.3, 0.4]
#     act = ['relu','linear','tanh']
#     nodes = [32, 64, 128]
#     layer_num= [2, 3, 4, 5, 6, 7]

#     return {'batch_size' : batches, 'optimizers' : optimizers, 'drop' : dropout, 'act': act, 'nodes' : nodes, 'layer' : layer_num}
# hyperparameters = create_hyperparameter()
# model2 = build_model()

# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# model2 = KerasClassifier(build_fn= build_model, verbose = 1)

# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
# # search = GridSearchCV(model2, hyperparameters, cv = 3)

# es = EarlyStopping(monitor = 'val_loss', patience= 5)
# lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.25)
# cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/keras61_{epoch:2d}_{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True)

# search.fit(x_train,y_train,verbose = 1, epochs = 100, validation_split = 0.2, callbacks = [es,cp,lr])

# print(search.best_params_)
# print(search.best_score_)
# acc = search.score(x_test, y_test)
# print('최종 스코어 : ', acc)


model = load_model('../data/modelcheckpoint/keras61_11_0.03.hdf5')
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)
print('체크포인트 스코어 : ', accuracy_score(y_test, y_pred))

# {'optimizers': 'adam', 'nodes': 32, 'layer': 2, 'drop': 0.4, 'batch_size': 32, 'act': 'relu'}
# 0.9796333312988281
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0708 - acc: 0.9849
# 최종 스코어 :  0.9848999977111816

# {'optimizers': 'adam', 'nodes': 64, 'layer': 5, 'drop': 0.4, 'batch_size': 64, 'act': 'relu'}
# 0.9906833370526632
# 157/157 [==============================] - 1s 5ms/step - loss: 0.0341 - acc: 0.9949
# 최종 스코어 :  0.9948999881744385
# 체크포인트 스코어 :  0.993