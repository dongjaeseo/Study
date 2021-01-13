import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys())  # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
print(dataset.values())

print(dataset.target_names) # ['setosa' 'versicolor' 'virginica']


# x = dataset.data
x = dataset['data']
y = dataset['target']
# y = dataset.target

print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,)
print(type(x), type(y)) # <class 'numpy.ndarray'>

df = pd.DataFrame(x, columns = dataset['feature_names']) # 넘파이를 판다스로 바꿔주는과정
print(df)
print(df.shape)
print(df.columns)
print(df.index)

print(df.head()) # 데이터셋 맨 처음 5개 출력 df[:5] 와 비슷
print(df.tail()) # 데이터셋 마지막 5개
print(df.info()) # info, 결측치 등을 알려줌
print(df.describe()) # 데이터셋의 수치값의 상세를 알려줌

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] # 컬럼명 갱신해줌
print(df.columns)
print(df.info())
print(df.describe())

# y칼럼을 추가하기
print(df['sepal_length'])
df['Target'] = dataset.target  # 타겟 칼럼에 y를 넣고 선언해주면 추가된다
print(df.head())

print(df.shape) # (150, 5)
print(df.columns)
print(df.index)
print(df.tail())

print(df.info())
print(df.isnull()) # 결측치가 있는지 확인
print(df.isnull().sum()) # 각 칼럼에 결측치가 몇개인지 보는방법
print(df.describe())
print(df['Target'].value_counts()) # df 의 타겟의 각 값이 몇개인지 세준다


# 상관계수 히트맵
print(df.corr()) # 피쳐들이 타겟과 얼마나 상관?있는지 확인

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale = 1.2)
# sns.heatmap(data=df.corr(), square = True, annot = True, cbar = True)
# plt.show()

# 도수 분포도
plt.figure(figsize =(10,6))
plt.subplot(2,2,1)
plt.hist(x ='sepal_length', data = df) # histogram 도수분포도 / df 라는 데이터프레임의 sepal length 라는 칼럼을 그릴거다
plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x ='sepal_width', data = df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x ='petal_length', data = df)
plt.title('petal_length')

plt.subplot(2,2,4)
plt.hist(x ='petal_width', data = df)
plt.title('petal_width')

plt.show()



