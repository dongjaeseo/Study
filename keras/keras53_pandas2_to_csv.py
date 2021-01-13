import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

# x = dataset.data
x = dataset['data']
# y = dataset.target
y = dataset['target']

df = pd.DataFrame(x, columns = dataset['feature_names']) # 넘파이를 판다스로 바꿔주는과정
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] # 컬럼명 갱신해줌

# y칼럼을 추가하기
df['Target'] = y # 타겟 칼럼에 y를 넣고 선언해주면 추가된다

df.to_csv('../data/csv/iris_sklearn.csv', sep =  ',')
