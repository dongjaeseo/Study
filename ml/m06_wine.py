import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

dataset = load_wine()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x,y, shuffle = True, train_size = 0.8)

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

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred,y_test)
    print('accuracy_score : ', accuracy)

# LinearSVC()
# model_score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# model_score :  1.0
# accuracy_score :  1.0
# model_score :  0.9722222222222222   
# accuracy_score :  0.9722222222222222

# SVC()
# model_score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# model_score :  1.0
# accuracy_score :  1.0
# model_score :  1.0
# accuracy_score :  1.0

# KNeighborsClassifier()
# model_score :  0.9166666666666666
# accuracy_score :  0.9166666666666666
# model_score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# model_score :  0.9444444444444444
# accuracy_score :  0.9444444444444444

# DecisionTreeClassifier()
# model_score :  0.8611111111111112
# accuracy_score :  0.8611111111111112
# model_score :  0.8888888888888888
# accuracy_score :  0.8888888888888888
# model_score :  0.9166666666666666
# accuracy_score :  0.9166666666666666

# RandomForestClassifier()
# model_score :  1.0
# accuracy_score :  1.0
# model_score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# model_score :  1.0
# accuracy_score :  1.0

# LogisticRegression()
# model_score :  1.0
# accuracy_score :  1.0
# model_score :  1.0
# accuracy_score :  1.0
# model_score :  0.9722222222222222
# accuracy_score :  0.9722222222222222