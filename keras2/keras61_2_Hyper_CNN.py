import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, ReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizers = Adam, act = 'relu', lr = 0.01, nodes = 256, layer = 1):
    optimizer = optimizers(lr = lr)
    inputs = Input(shape = (28,28,1), name = 'input')
    x = Conv2D(128, 3, activation = act, padding = 'same', name = 'conv1')(inputs)
    x = Dropout(drop)(x)
    for i in range(layer):
        x = Conv2D(nodes, 3, activation = act, padding = 'same', name = f'conv2{i}')(x)
    x = MaxPooling2D(3)(x)
    x = Conv2D(nodes, 5, activation = act, padding = 'same', name = 'conv3')(x)
    x = MaxPooling2D(5)(x)
    x = Flatten()(x)
    x = Dense(nodes, activation = act, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = act, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer, metrics = ['acc'], loss = 'categorical_crossentropy')

    return model

def create_hyperparameter():
    batches = [64, 128]
    optimizers = [Adam, RMSprop]
    lr = [0.01]
    dropout = [0.2, 0.3]
    act = ['relu']
    nodes = [256, 128]
    layer_num= [2, 3]

    return {'batch_size' : batches, 'optimizers' : optimizers, 'drop' : dropout, 'act': act, 'lr': lr, 'nodes' : nodes, 'layer' : layer_num}
hyperparameters = create_hyperparameter()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn= build_model, verbose = 1)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)

search.fit(x_train,y_train,verbose = 1)

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc)

# {'optimizers': <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>, 'lr': 0.01, 'drop': 0.2, 'batch_size': 128, 'act': 'relu'}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F888ACE5E0>
# 0.9522833426793417
# 79/79 [==============================] - 1s 9ms/step - loss: 0.1056 - acc: 0.9724
# 최종 스코어 :  0.9724000096321106

# {'optimizers': <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>, 'nodes': 256, 'lr': 0.01, 'layer': 2, 'drop': 0.2, 'batch_size': 128, 'act': 'relu'}       
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000218FEEA8820>
# 0.9395833214124044
# 79/79 [==============================] - 1s 14ms/step - loss: 0.3347 - acc: 0.9130
# 최종 스코어 :  0.9129999876022339

# {'optimizers': <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>, 'nodes': 256, 'lr': 0.01, 'layer': 2, 'drop': 0.2, 'batch_size': 128, 'act': 'relu'}       
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000028580ED8D00>
# 0.9078666567802429
# 79/79 [==============================] - 1s 14ms/step - loss: 0.1270 - acc: 0.9685
# 최종 스코어 :  0.968500018119812