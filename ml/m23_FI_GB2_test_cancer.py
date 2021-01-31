# 랜포
# 피처 임포턴스가25% 미만인 아덜 빼고

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8)

model1 = GradientBoostingClassifier(max_depth = 4)
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

model2 = GradientBoostingClassifier(max_depth = 4)
model2.fit(x_train,y_train)
print('\n컬럼 지우기 전 스코어!! : ', bef)
print('컬럼 지우고 난 스코어!! : ', model2.score(x_test,y_test))

plot_feature_importances_dataset(model2, x, feature)
plt.show()

# 각각의 피쳐 임포턴스!! :  [2.21610473e-05 2.72218296e-02 3.07926605e-04 1.44990287e-04
#  1.96159005e-02 5.87019319e-04 3.49126251e-03 3.59123913e-02
#  1.33192798e-04 1.28428164e-04 5.75722753e-03 2.02555027e-02
#  2.15101591e-03 3.39629936e-03 1.15268837e-03 2.82272617e-04
#  2.38130617e-03 5.66622569e-03 2.67413314e-04 4.13991469e-03
#  3.84536839e-01 3.04314486e-02 3.21571864e-01 2.22157875e-02
#  8.54370285e-03 1.03944548e-04 7.41619033e-03 8.94949666e-02
#  2.61431557e-03 5.59727437e-05]
# mean radius 는 하위 25퍼센트이므로 삭제!
# mean area 는 하위 25퍼센트이므로 삭제!
# mean symmetry 는 하위 25퍼센트이므로 삭제!
# mean fractal dimension 는 하위 25퍼센트이므로 삭제!
# compactness error 는 하위 25퍼센트이므로 삭제!
# symmetry error 는 하위 25퍼센트이므로 삭제!
# worst compactness 는 하위 25퍼센트이므로 삭제!
# worst fractal dimension 는 하위 25퍼센트이므로 삭제!

# 컬럼 지우기 전 스코어!! :  0.9473684210526315
# 컬럼 지우고 난 스코어!! :  0.9649122807017544