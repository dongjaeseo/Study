# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, plot_importance
import warnings
import datetime
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model1 = XGBClassifier(n_jobs = -1)
model1.fit(x_train,y_train)
bef = model1.score(x_test,y_test)

'''
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
'''

# plot_feature_importances_dataset(model1, dataset, 1)

plot_importance(model1)
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

model2 = XGBClassifier(n_jobs = -1)
model2.fit(x_train,y_train)
print('\n컬럼 지우기 전 스코어!! : ', bef)
print('컬럼 지우고 난 스코어!! : ', model2.score(x_test,y_test))

# plot_feature_importances_dataset(model2, x, feature)
# plt.show()

# 각각의 피쳐 임포턴스!! :  [0.00614176 0.08239651 0.64184142 0.26962031]
# sepal length (cm) 는 하위 25퍼센트이므로 삭제!

# 컬럼 지우기 전 스코어!! :  0.9666666666666667
# 컬럼 지우고 난 스코어!! :  1.0

#################################################################################################
# # 시간 확인용
# jobs = [-1, 1, 2, 4, 8]
# for i in jobs:
#     t1 = datetime.datetime.now()
#     model = XGBClassifier(n_jobs = i)
#     model.fit(x_train,y_train)
#     t2 = datetime.datetime.now()
#     print(f'\n{i} jobs 일때 걸린 시간은?!?! {t2-t1}')