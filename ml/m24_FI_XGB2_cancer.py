# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model1 = XGBClassifier(n_jobs = -1)
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

model2 = XGBClassifier(n_jobs = -1)
model2.fit(x_train,y_train)
print('\n컬럼 지우기 전 스코어!! : ', bef)
print('컬럼 지우고 난 스코어!! : ', model2.score(x_test,y_test))

# 각각의 피쳐 임포턴스!! :  [0.00728117 0.01588547 0.         0.         0.00423305 0.00484575
#  0.0235578  0.1423238  0.00207054 0.00483001 0.01689976 0.01157391
#  0.00266151 0.00728901 0.00492058 0.00313262 0.00057435 0.00978695
#  0.00358124 0.00252206 0.02673288 0.03552617 0.17909572 0.3201603
#  0.00691148 0.01122285 0.03188014 0.10669238 0.00363923 0.01016931]
# mean perimeter 는 하위 25퍼센트이므로 삭제!
# mean area 는 하위 25퍼센트이므로 삭제!
# mean symmetry 는 하위 25퍼센트이므로 삭제!
# perimeter error 는 하위 25퍼센트이므로 삭제!
# compactness error 는 하위 25퍼센트이므로 삭제!
# concavity error 는 하위 25퍼센트이므로 삭제!
# symmetry error 는 하위 25퍼센트이므로 삭제!
# fractal dimension error 는 하위 25퍼센트이므로 삭제!
# [10:14:20] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.

# 컬럼 지우기 전 스코어!! :  0.9912280701754386
# 컬럼 지우고 난 스코어!! :  0.9824561403508771