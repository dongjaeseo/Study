##가라로!!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

train2 = pd.read_csv('./mnist_1/train.csv')
test2 = pd.read_csv('./mnist_1/test.csv')

# train2 = train.drop(['id','digit','letter'],1)
# test2 = test.drop(['id','letter'],1)

# train2[train2<20] = 0
# test2[test2<20] = 0

train = pd.DataFrame(train2.loc[train2['letter'] == 'B', '0':].to_numpy()[0].reshape(28,28))
# train.to_csv('./mnist_1/alphabetb.csv')
df = pd.DataFrame(train)
print(df)
print(train2.iloc[0])

# x_train = train.iloc[:,3:].to_numpy()
# x_test = test.iloc[:,2:].to_numpy()
# # print(x_train.shape, x_test.shape) (2048, 784) (20480, 784)
# i = 'B'
# train_tmp = train.loc[train['letter'] == i, '0':]
# train_tmp = train_tmp.to_numpy()
# tmp = train_tmp[0].reshape(784,)
# tmp = np.sort(tmp)
# # train_tmp[0] = train_tmp[0].sort()
# print(tmp)
# pic = train_tmp[0].reshape(28,28)
# plt.imshow(pic)
# plt.show()

# df = pd.DataFrame(pic)
# df = df-192
# df[df < 0] = 0
# df[df > 0] = df[df > 0] + 192
# pic = df.to_numpy().reshape(28,28,1)

# plt.imshow(pic)
# plt.show()