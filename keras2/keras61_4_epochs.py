import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, RMSprop

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

#2. 모델
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def build_model(drop=0.5, optimizers = 'adam', act = 'relu', nodes = 64, layer = 2):
    inputs = Input(shape = (28*28,), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    for i in range(layer):
        x = Dense(nodes, activation = 'relu', name = f'hidden2_{i}')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizers, metrics = ['acc'], loss = 'categorical_crossentropy')

    return model

def create_hyperparameter():
    batches = [32, 64, 128]
    optimizers = ['adam', 'rmsprop']
    dropout = [0.2, 0.3, 0.4]
    act = ['relu','linear','tanh']
    nodes = [32, 64, 128]
    layer_num= [2, 3, 4, 5, 6, 7]

    return {'batch_size' : batches, 'optimizers' : optimizers, 'drop' : dropout, 'act': act, 'nodes' : nodes, 'layer' : layer_num}
hyperparameters = create_hyperparameter()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn= build_model, verbose = 1)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
# search = GridSearchCV(model2, hyperparameters, cv = 3)

es = EarlyStopping(monitor = 'val_loss', patience= 5)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.25)
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/keras61_{epoch:2d}_{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True)

search.fit(x_train,y_train,verbose = 1, epochs = 100, validation_split = 0.2, callbacks = [es,cp,lr])

print(search.best_params_)
print(search.best_score_)
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc)


# {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 40}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002D983E90E20>
# 0.9668500026067098
# 250/250 [==============================] - 0s 871us/step - loss: 0.1333 - acc: 0.9710
# 최종 스코어 :  0.9710000157356262