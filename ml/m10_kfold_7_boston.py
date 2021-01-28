import numpy as np
from sklearn.datasets import load_boston
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


dataset = load_boston()
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
# scores :  [0.64576921 0.73943608 0.79756562 0.65170904 0.64161898]

# DecisionTreeRegressor()
# scores :  [0.61613242 0.72689456 0.85126461 0.68220441 0.65602519]

# RandomForestRegressor()
# scores :  [0.86088704 0.86855496 0.87768879 0.55698601 0.85340749]

# LinearRegression()
# scores :  [0.66752292 0.65112804 0.75146493 0.79229414 0.52835078]