# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model1 = RandomForestClassifier(max_depth = 4)
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

model2 = RandomForestClassifier(max_depth = 4)
model2.fit(x_train,y_train)
print('\n컬럼 지우기 전 스코어!! : ', bef)
print('컬럼 지우고 난 스코어!! : ', model2.score(x_test,y_test))

plot_feature_importances_dataset(model2, x, feature)
plt.show()

# 각각의 피쳐 임포턴스!! :  [0.06252275 0.01599759 0.05331851 0.0472298  0.00730641 0.00472852
#  0.05606891 0.09440038 0.00232136 0.00383735 0.01809672 0.00114852
#  0.02022031 0.01268216 0.00197705 0.00213743 0.00307026 0.00581124
#  0.00170255 0.00378231 0.16730559 0.01204055 0.10355703 0.11057822
#  0.01453414 0.01244663 0.02657671 0.11564919 0.01423355 0.00471827]
# mean symmetry 는 하위 25퍼센트이므로 삭제!
# mean fractal dimension 는 하위 25퍼센트이므로 삭제!
# texture error 는 하위 25퍼센트이므로 삭제!
# smoothness error 는 하위 25퍼센트이므로 삭제!
# compactness error 는 하위 25퍼센트이므로 삭제!
# concavity error 는 하위 25퍼센트이므로 삭제!
# symmetry error 는 하위 25퍼센트이므로 삭제!
# fractal dimension error 는 하위 25퍼센트이므로 삭제!

# 컬럼 지우기 전 스코어!! :  0.9473684210526315
# 컬럼 지우고 난 스코어!! :  0.9736842105263158