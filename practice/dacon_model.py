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

#3. 모델
from tensorflow.keras.models import load_model
model = load_model('../dacon/data/modelcheckpoint/dacon.hdf5')

#4. 평가 예측
result = model.evaluate(x_test,y_test,batch_size = 8)
print(result)