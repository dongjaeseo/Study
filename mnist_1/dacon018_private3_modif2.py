import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential,Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from scipy import stats

train = pd.read_csv('./mnist_1/train_new.csv')
test = pd.read_csv('./mnist_1/test_new.csv')
sub = pd.read_csv('./mnist_1/submission.csv', header = 0)
train_ens = pd.read_csv('./mnist_1/train.csv')
test_ens = pd.read_csv('./mnist_1/test.csv')

# drop columns
train2 = train.drop(['id','digit','letter'],1)
train2_ens = train_ens.drop(['id','digit','letter'],1)

test2 = test.drop(['id','letter'],1)
test2_ens = test_ens.drop(['id','letter'],1)

# convert pandas dataframe to numpy array
train2 = train2.values
train2_ens = train2_ens.values
test2 = test2.values
test2_ens = test2_ens.values

# reshape
train2 = train2.reshape(-1,28,28,1)
train2_ens = train2_ens.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)
test2_ens = test2_ens.reshape(-1,28,28,1)

# data normalization
train2 = train2/255.0
train2_ens = train2_ens/255.0
test2 = test2/255.0
test2_ens = test2_ens/255.0

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# cross validation
skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)

reLR = ReduceLROnPlateau(patience=100,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=160, verbose=1)

result = 0
nth = 0

for train_index, valid_index in skf.split(train2,train['digit']) :
    
    mc = ModelCheckpoint('best_cvision.h5',save_best_only=True, verbose=1)
    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    x_train_ens = train2_ens[train_index]
    x_valid_ens = train2_ens[valid_index]
    
    train_generator = idg.flow([x_train,x_train_ens],y_train,batch_size=8)
    valid_generator = idg2.flow([x_valid,x_valid_ens],y_valid)
    test_generator = idg2.flow([test2,test2_ens],shuffle = False)
    
    input1 = Input(shape = (28,28,1))
    d1 = Conv2D(16,3,activation = 'relu', padding = 'same')(input1)
    d1 = BatchNormalization()(d1)
    d1 = Dropout(0.3)(d1)

    d1 = Conv2D(32, 3, activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1) 
    d1 = Conv2D(32, 5, activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Conv2D(32, 5, activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Conv2D(32, 5, activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = MaxPooling2D(3)(d1)
    d1 = Dropout(0.3)(d1)

    d1 = Conv2D(64, 3, activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Conv2D(64, 5, activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = MaxPooling2D(3)(d1)
    d1 = Dropout(0.3)(d1)

    d1 = Flatten()(d1)

    d1 = Dense(128, activation='relu')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Dropout(0.3)(d1)
    d1 = Dense(32, activation='relu')(d1)


    input2 = Input(shape = (28,28,1))
    d2 = Conv2D(16,3,activation = 'relu', padding = 'same')(input2)
    d2 = BatchNormalization()(d2)
    d2 = Dropout(0.3)(d2)

    d2 = Conv2D(32, 3, activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2) 
    d2 = Conv2D(32, 5, activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Conv2D(32, 5, activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Conv2D(32, 5, activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = MaxPooling2D(3)(d2)
    d2 = Dropout(0.3)(d2)

    d2 = Conv2D(64, 3, activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Conv2D(64, 5, activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = MaxPooling2D(3)(d2)
    d2 = Dropout(0.3)(d2)

    d2 = Flatten()(d2)

    d2 = Dense(128, activation='relu')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Dropout(0.3)(d2)
    d2 = Dense(32, activation='relu')(d2)

    d3 = concatenate([d1,d2])
    d3 = Dense(32, activation='relu')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.3)(d3)
    d3 = Dense(16, activation='relu')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.3)(d3)
    output = Dense(10, activation='softmax')(d3)

    model = Model(inputs = [input1,input2], outputs = output)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    
    learning_history = model.fit_generator(train_generator,epochs=10000, validation_data=valid_generator, callbacks=[es,mc,reLR])
    
    # predict
    model.load_weights('best_cvision.h5')
    result += model.predict_generator(test_generator,verbose=True)/40
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')

sub['digit'] = result.argmax(1)

sub.to_csv('./mnist_1/submission17.csv', index = 0)
