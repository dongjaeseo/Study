import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score ## 여기에요~!!!
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle = True, train_size = 0.8)

kfold = KFold(n_splits = 5, shuffle = True)

# from sklearn.preprocessing import MinMaxScaler
# scale = MinMaxScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

#2. modelling
model = RandomForestClassifier()

scores = cross_val_score(model, x_train, y_train, cv = kfold)
print('scores : ', scores)
# #3. compile fit
# model.fit(x_train,y_train)

# #4. evaluation, prediction
# result = model.score(x_test,y_test)
# print('model_score : ', result)
# y_pred = model.predict(x_test)
# acc = accuracy_score(y_pred,y_test)
# print('accuracy_score : ', acc)

# scores :  [1.         1.         0.91666667 0.875      0.95833333]