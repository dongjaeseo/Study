x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [2*l+3 for l in x]
# y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]

print(x, '\n', y)

import matplotlib.pyplot as plt
plt.plot(x, y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
print(df)
print(df.shape) # (10, 2)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
print(x_train.shape, y_train.shape) # (10,) (10,)
print(type(x_train)) # <class 'pandas.core.series.Series'>

x_train = x_train.values.reshape(len(x_train), 1)
print(x_train.shape, y_train.shape) # (10, 1) (10,)
print(type(x_train)) # <class 'numpy.ndarray'>

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print('score : ', score) # score :  1.0

print('기울기 : ', model.coef_) # [1.]
print('절편 : ', model.intercept_) # 1.0

# 기울기 :  [2.]       == 이 부분은 위에 y를 바꿈에 따라 같이 바뀐다!
# 절편 :  3.0
