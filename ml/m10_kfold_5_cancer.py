import numpy as np
from sklearn.datasets import load_breast_cancer
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


dataset = load_breast_cancer()
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
# scores :  [0.94505495 0.96703297 0.94505495 0.98901099 0.98901099]

# SVC()
# scores :  [0.94505495 0.97802198 0.97802198 0.95604396 0.96703297]

# KNeighborsClassifier()
# scores :  [0.93406593 0.95604396 0.96703297 0.98901099 0.96703297]

# DecisionTreeClassifier()
# scores :  [0.91208791 0.94505495 0.9010989  0.87912088 0.96703297]

# RandomForestClassifier()
# scores :  [0.95604396 0.94505495 0.94505495 0.95604396 1.        ]

# LogisticRegression()
# scores :  [1.         0.96703297 1.         0.97802198 0.96703297]