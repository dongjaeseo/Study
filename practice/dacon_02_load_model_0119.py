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
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data


data = Add_features(df)
print(data[:48])

data = df.to_numpy()
data = data.reshape(1095,48,10)

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

timestep = 4
x,y = split_xy(data,timestep,2)


from sklearn.model_selection import train_test_split as tts
x_train,x_val,y_train,y_val = tts(x,y,train_size = 0.8, shuffle = True)

#2. 모델구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU, Reshape

# 모델 9번 돌리기 
d = []
for l in range(9):
    model = load_model(filepath = '../dacon/data/modelcheckpoint/dacon_ts%d_%d_0119.hdf5'%(timestep,l))

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

