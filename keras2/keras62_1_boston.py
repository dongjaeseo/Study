import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense,Input, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import r2_score

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, shuffle = True)

scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
# print(x_train.shape) #(404, 13)

#2. 모델링
def build(act = 'relu', hidden = 64, hidden_num = 2, drop = 0.2, optimizer = 'adam', batch_size = 32):
    batch_size = batch_size
    inputs = Input(shape = (13,))
    x = Dense(128, activation = act)(inputs)
    for i in range(hidden_num):
        x = Dense(hidden, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(1)(x)
    model = Model(inputs = inputs, outputs = x)
    model.compile(loss = 'mse', optimizer= optimizer, metrics = ['mae'])
    return model

def param():
    activation = ['relu']
    hidden = [64, 128, 256, 512]
    hidden_num = [9,11,13]
    drop = [0.2, 0.3]
    batches = [8,16,32]
    optimizers = ['adam']
    return {'act': activation, 'hidden' : hidden, 'hidden_num' : hidden_num, 'drop' : drop, 'batch_size' : batches, 'optimizer': optimizers}

model = KerasRegressor(build_fn = build, verbose = 1)
parameters = param()

search = RandomizedSearchCV(model, parameters, cv = 4)

es = EarlyStopping(monitor = 'val_loss', patience = 5)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.25)
cp = ModelCheckpoint(monitor = 'val_loss', filepath = '../data/modelcheckpoint/keras62_{epoch:2d}_{val_loss:.4f}.hdf5', save_best_only=True)

search.fit(x_train, y_train, epochs = 1000, validation_split = 0.2, callbacks = [es,lr,cp])

print(search.best_params_)
print('최종점수 : ', search.score(x_test,y_test))

# search = load_model('../data/modelcheckpoint/keras62_20_17.2824.hdf5')

y_pred = search.predict(x_test)
print('R2 : ', r2_score(y_test, y_pred))

# R2 :  0.8890386401784056