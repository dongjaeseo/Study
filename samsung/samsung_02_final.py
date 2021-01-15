import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_x(seq,size):
    a = []
    for i in range(len(seq)-size+1):
        subset = seq[i:i+size]
        a.append(subset)
    return np.array(a)  



'''

# 액면분할
df.iloc[662+days:, :4] /= 50.
df.iloc[662+days:,[4]] *= 50.
df['target'] = df.iloc[:,[3]]
df = df.astype(float)
df.dropna(inplace = True)
dataset = df.to_numpy()

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(dataset[:, :-1]) # 종가값 y 를 제외한 값들을 전처리해준다
dataset[:, :-1] = scale.transform(dataset[:, :-1])

df = pd.DataFrame(dataset)

size = 10
df_x = df.iloc[1:, :-1] # 상관계수 높은 7개만 사용
df_y = df.iloc[:-(size), [-1]]
df_x_pred = df.iloc[0:size+8, :-1]

df_x = df_x.iloc[::-1]
df_y = df_y.iloc[::-1]
df_x_pred = df_x_pred.iloc[::-1] # 데이터 순서를 거꾸로 해준다

x = df_x.to_numpy().astype(float)
y = df_y.to_numpy().astype(float)
x_pred = df_x_pred.to_numpy().astype(float)

x = split_x(x,size)
x_pred = split_x(x_pred,size)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, shuffle = True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8, shuffle = True)
'''