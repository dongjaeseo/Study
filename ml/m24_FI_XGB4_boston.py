# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model1 = XGBRegressor(n_jobs = -1)
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

# plot_feature_importances_dataset(model1, dataset, 1)
# plt.show()

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

model2 = XGBRegressor(n_jobs = -1)
model2.fit(x_train,y_train)
print('\n컬럼 지우기 전 스코어!! : ', bef)
print('컬럼 지우고 난 스코어!! : ', model2.score(x_test,y_test))

# 각각의 피쳐 임포턴스!! :  [1.7965430e-02 2.0223067e-03 1.7186860e-02 3.8977052e-04 5.2890379e-02
#  4.1205403e-01 6.5909489e-03 3.7562620e-02 2.0892795e-02 2.3242185e-02
#  3.1394839e-02 7.2638043e-03 3.7054405e-01]
# ZN 는 하위 25퍼센트이므로 삭제!
# CHAS 는 하위 25퍼센트이므로 삭제!
# AGE 는 하위 25퍼센트이므로 삭제!

# 컬럼 지우기 전 스코어!! :  0.8287638565874602
# 컬럼 지우고 난 스코어!! :  0.8994886867788905