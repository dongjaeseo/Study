import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def ens_model():
    input1 = Input(shape = (28,28,1))
    d = Conv2D(16,(3,3),activation='relu',padding='same')(input1)
    d = BatchNormalization()(d)
    d = Dropout(0.3)(d)
    d = Conv2D(32,(3,3),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2D(32,(5,5),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    # d = Conv2D(32,(5,5),activation='relu',padding='same')(d)
    # d = BatchNormalization()(d)
    # d = Conv2D(32,(5,5),activation='relu',padding='same')(d)
    # d = BatchNormalization()(d)
    d = MaxPooling2D(3,3)(d)
    d = Dropout(0.3)(d)

    d = Conv2D(64,(3,3),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2D(64,(5,5),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = MaxPooling2D(3,3)(d)
    d = Dropout(0.3)(d)
    
    d = Flatten()(d)

    d = Dense(128,activation='relu')(d)
    d = BatchNormalization()(d)
    d = Dropout(0.3)(d)
    d = Dense(64,activation='relu')(d)
    d = BatchNormalization()(d)
    d1 = Dense(32,activation='relu')(d)


    input2 = Input(shape = (28,28,1))
    d = Conv2D(16,(3,3),activation='relu',padding='same')(input2)
    d = BatchNormalization()(d)
    d = Dropout(0.3)(d)
    d = Conv2D(32,(3,3),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2D(32,(5,5),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2D(32,(5,5),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2D(32,(5,5),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = MaxPooling2D(3,3)(d)
    d = Dropout(0.3)(d)

    d = Conv2D(64,(3,3),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2D(64,(5,5),activation='relu',padding='same')(d)
    d = BatchNormalization()(d)
    d = MaxPooling2D(3,3)(d)
    d = Dropout(0.3)(d)
    
    d = Flatten()(d)

    d = Dense(128,activation='relu')(d)
    d = BatchNormalization()(d)
    d = Dropout(0.3)(d)
    d = Dense(64,activation='relu')(d)
    d = BatchNormalization()(d)
    d2 = Dense(32,activation='relu')(d)

    new_d = concatenate([d1,d2])
    d3 = Dense(32,activation = 'relu')(new_d)
    d3 = BatchNormalization()(d3)    
    d3 = Dense(32,activation = 'relu')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Dense(32,activation = 'relu')(d3)
    d3 = Dense(10, activation = 'softmax')(d3)
    
    model = Model(inputs= [input1,input2], outputs = d3)
    return model

train = pd.read_csv('./mnist_1/train.csv')
test = pd.read_csv('./mnist_1/test.csv')
submission = pd.read_csv('./mnist_1/submission.csv', header = 0)
submission['tmp'] = test['letter']

alphabets = string.ascii_uppercase
alphabets = list(alphabets)

skf = StratifiedKFold(n_splits=15, random_state=33, shuffle=True)
datagen = ImageDataGenerator(width_shift_range= 0.2, height_shift_range= 0.2)
datagen2 = ImageDataGenerator()
es = EarlyStopping(monitor = 'val_loss', patience = 30)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.25, patience = 10)


x1_train = train.loc[:, '0':].to_numpy().reshape(-1,28,28,1)
y1_train = train.loc[:, 'digit'].to_numpy()
x_test = test.loc[:, '0':].to_numpy().reshape(-1,28,28,1)

ens_x1_train = train.loc[:, '0':]
ens_x1_train[ens_x1_train<192] /= 2
ens_x1_train = ens_x1_train.to_numpy().reshape(-1,28,28,1)

ens_x_test = test.loc[:, '0':]
ens_x_test[ens_x_test<192] /= 2
ens_x_test = ens_x_test.to_numpy().reshape(-1,28,28,1)

cp = ModelCheckpoint(monitor= 'val_loss', filepath=f'../dacon_1/models/all.h5', save_best_only= True)

result = []
for train_index, valid_index in skf.split(x1_train, y1_train, ens_x1_train) :
    x_train = x1_train[train_index]
    x_val = x1_train[valid_index]    
    y_train = y1_train[train_index]
    y_val = y1_train[valid_index]
    ens_x_train = ens_x1_train[train_index]
    ens_x_val = ens_x1_train[valid_index]

    train_generator = datagen.flow([x_train,ens_x_train],y_train,batch_size=8)
    val_generator = datagen2.flow([x_val,ens_x_val],y_val)

    model = ens_model()
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model.fit_generator(train_generator, epochs = 1000, validation_data= val_generator, callbacks = [es,cp,lr])

    model.load_weights(f'../dacon_1/models/all.h5')
    y_pred = model.predict([x_test,ens_x_test])
    y_pred = y_pred.argmax(1)
    result.append(y_pred)
result = np.array(result)
mode = stats.mode(result).mode
mode = np.transpose(mode)
submission.loc[:, 'digit'] = mode

submission.drop('tmp', axis = 1,inplace=True)
submission.to_csv('./mnist_1/submission11.csv', index = 0)

count = stats.mode(result).count
for i in count[0]:
    print(i)