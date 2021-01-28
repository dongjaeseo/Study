import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score ## 여기에요~!!!
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


dataset = load_wine()
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
models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i
    print(f'\n{i}')
    scores = cross_val_score(model, x_train, y_train, cv = kfold)
    print('scores : ', scores)

# LinearSVC()
# scores :  [0.96551724 0.93103448 1.         1.         0.96428571]

# SVC()
# scores :  [1.         1.         1.         0.96428571 1.        ]

# KNeighborsClassifier()
# scores :  [1.         0.93103448 1.         0.96428571 0.96428571]

# DecisionTreeClassifier()
# scores :  [0.96551724 0.93103448 0.92857143 0.96428571 0.78571429]

# RandomForestClassifier()
# scores :  [1.         1.         0.96428571 0.96428571 0.96428571]

# LogisticRegression()
# scores :  [1.         0.96551724 1.         0.96428571 1.        ]