import numpy as np
import pandas as pd

train = pd.read_csv('./practice/dacon/data/train/train.csv')
submission = pd.read_csv('./practice/dacon/data/sample_submission.csv')

day = 4

def split_to_seq(data):
    tmp = []
    for i in range(48):
        tmp1 = pd.DataFrame()
        for j in range(int(len(data)/48)):
            tmp2 = data.iloc[j*48+i,:]
            tmp2 = tmp2.to_numpy()
            tmp2 = tmp2.reshape(1,tmp2.shape[0])
            tmp2 = pd.DataFrame(tmp2)
            tmp1 = pd.concat([tmp1,tmp2])
        x = tmp1.to_numpy()
        tmp.append(x)
    return np.array(tmp)

def make_cos(dataframe): # 특정 열이 해가 뜨고 해가지는 시간을 가지고 각 시간의 cos를 계산해주는 함수
    dataframe /=dataframe
    c = dataframe.dropna()
    d = c.to_numpy()

    def into_cosine(seq):
        for i in range(len(seq)):
            if i < len(seq)/2:
                seq[i] = float((len(seq)-1)/2) - (i)
            if i >= len(seq)/2:
                seq[i] = seq[len(seq) - i - 1]
        seq = seq/ np.max(seq) * np.pi/2
        seq = np.cos(seq)
        return seq

    d = into_cosine(d)
    dataframe = dataframe.replace(to_replace = np.NaN, value = 0)
    dataframe.loc[dataframe['cos'] == 1] = d
    return dataframe

def preprocess_data(data, is_train = True):
    a = pd.DataFrame()
    for i in range(int(len(data)/48)):
        tmp = pd.DataFrame()
        tmp['cos'] = data.loc[i*48:(i+1)*48-1,'TARGET']
        tmp['cos'] = make_cos(tmp)
        a = pd.concat([a,tmp])
    data['cos'] = a
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.insert(1,'Time',data['Hour']*2+data['Minute']/30.)
    temp = data.copy()
    temp = temp[['Time','TARGET','GHI','DHI','DNI','WS','RH','T']]

    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        temp = temp[['Time','TARGET','GHI','DHI','DNI','WS','RH','T']]
        return temp.iloc[-48*day:, :]

df_train = preprocess_data(train)

df_test = []
for i in range(81):
    file_path = './practice/dacon/data/test/%d.csv'%i
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)

train = split_to_seq(df_train)
test = split_to_seq(x_test)

# print(train.shape)(48, 1093, 10)
# print(test.shape) #(48, 324, 8)