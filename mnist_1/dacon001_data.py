import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

train = pd.read_csv('./mnist_1/train.csv')
test = pd.read_csv('./mnist_1/test.csv')
# print(train.shape, test.shape) (2048, 787) (20480, 786)

alphabets = string.ascii_uppercase
alphabets = list(alphabets)

### 각 알파벳 별로 몇개의 숫자씩 있는지 확인하는중
### 목표는 같은 알파벳끼리 묶고, 거기서 같은 숫자들끼리 묶으려는데
### 한 알파벳에 있는 숫자의 데이터양이 적으면 안되니 그걸 확인해보려한다
'''
for i in alphabets:
    train_tmp = train.loc[train['letter'] == i, ['digit','letter']]
    for j in range(10):
        tmp = train_tmp.loc[train_tmp['digit'] == j]
        # if len(tmp) < 5:
        print(f'{i} 알파벳에 {j} 는 {len(tmp)}개!')
'''

### 근데 한 알파벳에 5개도 안되는 숫자들이 꽤 많다!!
### A 알파벳에 3 는 1개! 이런거??

### 그럼 시각화를 해보자
x_train = train.iloc[:,3:].to_numpy()
x_test = test.iloc[:,2:].to_numpy()
# print(x_train.shape, x_test.shape) (2048, 784) (20480, 784)
i = 'B'
train_tmp = train.loc[train['letter'] == i, '0':]
train_tmp = train_tmp.to_numpy()
tmp = train_tmp[0].reshape(784,)
tmp = np.sort(tmp)
# train_tmp[0] = train_tmp[0].sort()
print(tmp)
pic = train_tmp[0].reshape(28,28)
plt.imshow(pic)
plt.show()

df = pd.DataFrame(pic)
df = df-192
df[df < 0] = 0
df[df > 0] = df[df > 0] + 192
pic = df.to_numpy().reshape(28,28,1)

plt.imshow(pic)
plt.show()
# j = 0
# while True:    
#     pic = train_tmp[j].reshape(28,28,1)
#     plt.imshow(pic)
#     plt.show()
#     j+= 1
#     if j > len(train_tmp):
#         break

# for i in range(5):
#     pic = x_train[i].reshape(28,28,1)
#     plt.imshow(pic)
#     plt.show()
