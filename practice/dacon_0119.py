import numpy as np
import pandas as pd

df = pd.read_csv('./practice/dacon/data/train/train.csv')

df.drop(['Minute','Day'], axis =1, inplace = True)
# print(df.shape) # (52560, 7)

def Add_features(data):
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data.insert(1,'GHI',data['DNI']+data['DHI'])
    return data

data = Add_features(df)
data = df.to_numpy()
data = data.reshape(1095,48,10)

# for i in range(81):
#     df = pd.read_csv('./practice/dacon/data/test/%d.csv'%i,index_col= None, header = 0)
#     df.drop(['Minute','Day'], axis =1, inplace = True)
#     df = Add_features(df)
#     df.to_csv('./practice/dacon/data/newtest/%d.csv'%i, index = None)

def split_xy(data,timestep,ynum):
    x,y = [],[]
    for i in range(len(data)):
        x_end = i + timestep
        y_end = x_end + ynum
        if y_end > len(data):
            break
        x_tmp = data[i:x_end]
        y_tmp = data[x_end:y_end,:,-1]
        x.append(x_tmp)
        y.append(y_tmp)
    return(np.array(x),np.array(y))

timestep = 1
x,y = split_xy(data,timestep,2)


from sklearn.model_selection import train_test_split as tts
x_train,x_val,y_train,y_val = tts(x,y,train_size = 0.8, shuffle = True, random_state = 0)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU, Reshape

drop = 0.3
model = Sequential()
model.add(Conv2D(1024,2,padding = 'same', input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(drop))
model.add(Conv2D(512,2,padding = 'same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(drop))
model.add(Conv2D(256,2,padding = 'same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(Conv2D(128,2,padding = 'same'))
model.add(LeakyReLU(alpha = 0.05))
model.add(Flatten())
model.add(Dense(4096))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(drop))
model.add(Dense(2048))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(1024))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(512))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(256))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(2*48))
model.add(Reshape((2,48)))
# model.summary()

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 10, verbose = 1)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# 모델 9번 돌리기 
d = []
for l in range(9):
    cp = ModelCheckpoint(filepath = '../dacon/data/modelcheckpoint/dacon_ts:%d_%d_0119.hdf5'%(timestep,l),monitor='val_loss', save_best_only=True)
    model.fit(x,y,epochs= 1, validation_split=0.2, batch_size =8, callbacks = [es,cp,lr])

    c = []
    for i in range(81):
        testx = pd.read_csv('./practice/dacon/data/test/%d.csv'%i)
        testx.drop(['Minute','Day'], axis =1, inplace = True)
        testx = Add_features(testx)
        testx = (testx.copy()).iloc[-48*timestep:]
        testx = testx.to_numpy()
        testx = testx.reshape(timestep,48,10)
        testx,null_y = split_xy(testx,timestep,0)
        y_pred = model.predict(testx)
        y_pred = y_pred.reshape(2,48)
        a = []
        for j in range(2):
            b = []
            for k in range(48):
                b.append(y_pred[j,k])
            a.append(b)
        c.append(a)
    d.append(c)
d = np.array(d)
# print(d.shape) (9, 81, 2, 48)


### 뻘짓!! 쉐이프 바꿔주는중~~~
e = []
for i in range(81):
    f = []
    for j in range(2):
        g = []
        for k in range(48):
            h = []
            for l in range(9):
                h.append(d[l,i,j,k])
            g.append(h)
        f.append(g)
    e.append(f)

e = np.array(e)
df_sub = pd.read_csv('./practice/dacon/data/sample_submission.csv', index_col = 0, header = 0)

# submit 파일에 데이터들 덮어 씌우기!!
for i in range(81):
    for j in range(2):
        for k in range(48):
            df = pd.DataFrame(e[i,j,k])
            df[df < 0] = 0
            for l in range(9):
                x = df.quantile(q = ((l+1)/10.),axis = 0)[0]
                df_sub.iloc[[i*96+j*48+k],[l]] = x

df_sub.to_csv('./practice/dacon/data/submit_0119.csv')


