# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model1 = RandomForestClassifier(max_depth = 4)
model1.fit(x_train,y_train)
print('컬럼 지우기 전 스코어!! : ', model1.score(x_test,y_test))

def plot_feature_importances_dataset(model, dataset, feature):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    try:
        plt.yticks(np.arange(n_features), dataset.feature_names)
    except:
        plt.yticks(np.arange(n_features), feature)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model1, dataset, 1)
# plt.show()

#################################################################################################
columns = model1.feature_importances_
print('각각의 피쳐 임포턴스!! : ', columns)
print(dataset.feature_names)
quantile = pd.DataFrame(columns).quantile(q = 0.25)
df = pd.DataFrame(x).copy()

j = 0
numbers = []
for i in columns:
    if i < quantile[0]:
        print(f'{dataset.feature_names[j]} 는 하위 25퍼센트이므로 삭제!')
        numbers.append(j)
    j += 1
    
df = df.drop(df.columns[numbers], axis =1)
print(df)
print(dataset.data)