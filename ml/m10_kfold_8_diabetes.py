import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score ## 여기에요~!!!
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


dataset = load_diabetes()
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
models = [KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), LinearRegression()]
for i in models:
    model = i
    print(f'\n{i}')
    scores = cross_val_score(model, x_train, y_train, cv = kfold)
    print('scores : ', scores)

# KNeighborsRegressor()
# scores :  [0.34942819 0.43793936 0.35369965 0.41513786 0.41801545]

# DecisionTreeRegressor()
# scores :  [-0.22026757  0.12070525 -0.57705632  0.17638698 -0.21107687]

# RandomForestRegressor()
# scores :  [0.31165559 0.47671358 0.5952994  0.17684229 0.58024772]

# LinearRegression()
# scores :  [0.5683243  0.41925247 0.46179818 0.56143612 0.29447092]