# train test 나눈 다음에 train만 발리데이션 하지 말고, 
# kfold 한 후에 train_test_split 사용

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

kfold = KFold(n_splits = 5, shuffle = True)

#2. modelling
model = RandomForestClassifier()
for train_idx, val_idx in kfold.split(x,y): 
    # train fold, val fold 분할
    x_train = x[train_idx]
    x_test = x[train_idx]
    y_train = y[train_idx]
    y_test = y[train_idx]

    scores = cross_val_score(model, x_train, y_train, cv = kfold)
    print('score : ', scores)

# score :  [0.91666667 0.91666667 0.91666667 0.95833333 0.95833333]
# score :  [1.         1.         0.875      0.95833333 0.91666667]
# score :  [0.91666667 0.95833333 0.95833333 0.95833333 1.        ]
# score :  [0.95833333 1.         0.91666667 1.         0.875     ]
# score :  [0.91666667 0.95833333 1.         0.95833333 1.        ]