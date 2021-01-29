from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
dataset = load_boston()
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,train_size = 0.8, random_state = 33)

#2. 모델
model = DecisionTreeRegressor(max_depth = 4)

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
  
# [0.05730848 0.         0.         0.         0.00697375 0.63859682
#  0.         0.09181176 0.         0.         0.00493084 0.        
#  0.20037834]
# acc :  0.6604355946809266
# RM 이 중요!!