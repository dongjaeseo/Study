from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
print('acc 칼럼 지우기전!! : ', acc)

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

plot_feature_importances_dataset(model,dataset,1)
plt.show()

####### 여기까지가 기존!!

####### 피처임포턴스의 길이인 30 동안 fi가 0이 아닐때마다 데이터셋에서 칼럼을 끌어와서 new_data 에 넣어줌!!!!
####### 이때 피쳐라는 리스트에 피처네임을 더해줌!!!
##### >> new_data 에는 (n,569) >> transpose 해주면 (569,n) 의 데이터가 되고
##### >> 각각에 대응하는 피처네임은 피처라는 리스트에 더해줌!!

#### 넘파이 사용!!!
fi = model.feature_importances_
new_data = []
feature = []
for i in range(len(fi)):
    if fi[i] != 0:
        new_data.append(dataset.data[:,i])
        feature.append(dataset.feature_names[i])
new_data = np.array(new_data)
new_data = np.transpose(new_data)



#### 데이터프레임 사용!!!
# fi = model.feature_importances_
# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# numbers = []
# for i in range(len(fi)):
#     if fi[i] == 0:
#         numbers.append(i)
# df = df.drop(df.columns[numbers], axis = 1)
# new_data = df.copy()
# feature = new_data.columns


x_train2,x_test2,y_train2,y_test2 = train_test_split(new_data,dataset.target,train_size = 0.8, random_state = 33)

#2. 모델
model2 = DecisionTreeRegressor(max_depth = 4)

#3. 훈련
model2.fit(x_train2, y_train2)

#4. 평가 예측
acc2 = model2.score(x_test2,y_test2)

print(model2.feature_importances_)
print('acc 칼럼 지우고!!!! : ', acc2)

####### dataset >> new_data 로 바꾸고 featurename 부분을 feature 리스트로 바꿔줌!!!
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

plot_feature_importances_dataset(model2,new_data,feature)
plt.show()

# [0.04032715 0.         0.         0.         0.02383042 0.63787124
#  0.         0.09181176 0.         0.         0.00565641 0.
#  0.20050301]
# acc 칼럼 지우기전!! :  0.6604355946809266
# [0.04032715 0.02383042 0.63787124 0.09266201 0.00493084 0.20037834]
# acc 칼럼 지우고!!!! :  0.6892183619607892