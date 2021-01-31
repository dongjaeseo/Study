# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model1 = RandomForestRegressor(max_depth = 4)
model1.fit(x_train,y_train)
bef = model1.score(x_test,y_test)

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

plot_feature_importances_dataset(model1, dataset, 1)
plt.show()

#################################################################################################

columns = model1.feature_importances_
print('각각의 피쳐 임포턴스!! : ', columns)
quantile = pd.DataFrame(columns).quantile(q = 0.25)
df = pd.DataFrame(x, columns = dataset.feature_names).copy()

j = 0
numbers = []
for i in columns:
    if i < quantile[0]:
        print(f'{dataset.feature_names[j]} 는 하위 25퍼센트이므로 삭제!')
        numbers.append(j)
    j += 1
    
new_data = df.drop(df.columns[numbers], axis =1)
feature = new_data.columns
new_data = new_data.to_numpy()

#################################################################################################

x = new_data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model2 = RandomForestRegressor(max_depth = 4)
model2.fit(x_train,y_train)
print('\n컬럼 지우기 전 스코어!! : ', bef)
print('컬럼 지우고 난 스코어!! : ', model2.score(x_test,y_test))

plot_feature_importances_dataset(model2, x, feature)
plt.show()

# 각각의 피쳐 임포턴스!! :  [3.22326134e-02 2.58461089e-04 1.65182116e-03 7.29747153e-04
#  2.19978029e-02 3.33432578e-01 9.21266472e-03 6.29232573e-02
#  1.75378221e-03 8.79159657e-03 1.47244958e-02 5.40059335e-03
#  5.06890586e-01]
# ZN 는 하위 25퍼센트이므로 삭제!
# INDUS 는 하위 25퍼센트이므로 삭제!
# CHAS 는 하위 25퍼센트이므로 삭제!

# 컬럼 지우기 전 스코어!! :  0.892900417431785
# 컬럼 지우고 난 스코어!! :  0.8764586031128558
