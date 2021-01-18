import numpy as np
import pandas as pd

df = pd.read_csv('./practice/dacon/data/train/train.csv')

df.drop(['Hour','Minute','Day'], axis =1, inplace = True)
# print(df.shape) # (52560, 7)

data = df.to_numpy()
data = data.reshape(1095,48,6)

def split_xy(data,timestep,ynum):
    x,y = [],[]
    for i in range(len(data)):
        x_end = i + timestep
        y_end = x_end + ynum
        if y_end > len(data):
            break
        x_tmp = data[i:x_end]
        y_tmp = data[x_end:y_end]
        x.append(x_tmp)
        y.append(y_tmp)
    return(np.array(x),np.array(y))

x,y = split_xy(data,7,2)
# x.shape = (1087,7,48,6)
# y.shape = (1087,2,48,6)



from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,train_size = 0.8, shuffle = True, random_state = 0)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU, Reshape

drop = 0.3
model = Sequential()
model.add(Conv2D(512,2,padding = 'same', input_shape = (7,48,6)))
model.add(LeakyReLU(alpha = 0.05))
model.add(MaxPooling2D(2))
model.add(Dropout(drop))
model.add(Conv2D(256,2,padding = 'same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(drop))
model.add(Conv2D(128,2,padding = 'same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(drop))
model.add(Dense(128))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(256))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(2*48*6))
model.add(LeakyReLU(alpha = 0.05))
model.add(Reshape((2,48,6)))
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 20)
cp = ModelCheckpoint(filepath = '../dacon/data/modelcheckpoint/dacon.hdf5',monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 10, verbose = 1)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train,y_train,epochs= 1000, validation_split=0.2, batch_size =8, callbacks = [es,cp,lr])

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 8)
print(result)
