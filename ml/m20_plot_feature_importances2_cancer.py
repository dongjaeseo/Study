from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
dataset = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,train_size = 0.8, random_state = 33)

#2. 모델
model = DecisionTreeClassifier(max_depth = 4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)
print('acc : ', acc)

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()
    
# [0.         0.         0.         0.         0.         0.
#  0.         0.02806298 0.0077365  0.         0.         0.
#  0.         0.         0.00149978 0.         0.         0.
#  0.         0.00947765 0.0470051  0.02901187 0.75233436 0.
#  0.         0.         0.06201816 0.0628536  0.         0.        ]
#  acc :  0.8947368421052632
# worst perimeter 가 0.75