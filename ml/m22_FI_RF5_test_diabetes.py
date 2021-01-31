# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = load_diabetes()
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

# 각각의 피쳐 임포턴스!! :  [0.02574574 0.0028516  0.31879499 0.09824793 0.02033837 0.01725924
#  0.03748756 0.0109452  0.40766685 0.06066253]
# sex 는 하위 25퍼센트이므로 삭제!
# s2 는 하위 25퍼센트이므로 삭제!
# s4 는 하위 25퍼센트이므로 삭제!

# 컬럼 지우기 전 스코어!! :  0.3942181428428605
# 컬럼 지우고 난 스코어!! :  0.5016489325447364