import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV ## 여기에요~!!!
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train,x_test, y_train,y_test = train_test_split(x,y,train_size = 0.8)

kfold = KFold(n_splits = 5, shuffle = True)

parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma": [0.001, 0.0001]},
    {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma": [0.001, 0.0001]},
] ## SVC 에 들어가는 파라미터를 조정해주는것

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(x)
x = scale.transform(x)

#2. modelling
model = GridSearchCV(SVC(), parameters, cv = kfold)
# scores = cross_val_score(model, x_train, y_train, cv = kfold)

# #3. compile fit
model.fit(x_train,y_train)

# #4. evaluation, prediction
print('최적의 매개변수 : ', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률', accuracy_score(y_test,y_pred))
print('최종정답률', model.score(x_test,y_test))