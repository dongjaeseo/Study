import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

def mymodel():
    model = Sequential()
    model.add(Conv2D(256, 3, padding = 'same', activation = 'relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(256, 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(128, 3, padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    return model

train = pd.read_csv('./mnist_1/train.csv')
test = pd.read_csv('./mnist_1/test.csv')
submission = pd.read_csv('./mnist_1/submission.csv', header = 0)

alphabets = string.ascii_uppercase
alphabets = list(alphabets)


for alphabet in alphabets:
    x_train = train.loc[train['letter'] == alphabet, '0':].to_numpy().reshape(-1,28,28,1)
    y_train = train.loc[train['letter'] == alphabet, 'digit'].to_numpy()
    x_test = test.loc[test['letter'] == alphabet, '0':].to_numpy().reshape(-1,28,28,1)


    y_train = to_categorical(y_train)
    generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.05, height_shift_range= 0.05)
    data = generator.flow(x_train,y_train,batch_size = 32)
    model = mymodel()

    filepath = f'../dacon_1/models/{alphabet}.hdf5'
    es = EarlyStopping(monitor = 'loss', patience= 20)
    lr = ReduceLROnPlateau(monitor= 'loss', patience = 10, factor = 0.25, verbose = 1)
    cp = ModelCheckpoint(filepath=filepath,monitor = 'loss', save_best_only= True)
    model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['acc'])
    model.fit_generator(data,epochs = 1000, callbacks = [es,cp,lr])

'''
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    #2. 모델링
    model = mymodel()

    #3. 컴파일 훈련
    filepath = f'../dacon_1/models/{alphabet}.hdf5'
    es = EarlyStopping(monitor = 'val_loss', patience= 20)
    lr = ReduceLROnPlateau(monitor= 'val_loss', patience = 10, factor = 0.25, verbose = 1)
    cp = ModelCheckpoint(filepath=filepath,monitor = 'val_loss', save_best_only= True)

    model.compile(loss = 'categorical_crossentropy', optimizer= 'adam')
    model.fit(x_train,y_train,batch_size = 32, epochs = 10000, validation_data= (x_val,y_val), callbacks = [es,cp,lr])
    
'''