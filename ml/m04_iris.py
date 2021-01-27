import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = load_iris()
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
model = RandomForestClassifier()

#3. compile fit
model.fit(x_train,y_train)

#4. evaluation, prediction
result = model.score(x_test,y_test)
print('model_score : ', result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)
print('accuracy_score : ', acc)

######## MinMaxScaler ###############################################
# Linear 빼고 얼추 다 비슷한듯 #######################################

#1. LinearSVC 
# model_score :  0.8666666666666667
# accuracy_score :  0.8666666666666667

#2. SVC
# model_score :  0.9333333333333333
# accuracy_score :  0.9333333333333333

#3. KNeighboursClassifier
# model_score :  0.9333333333333333
# accuracy_score :  0.9333333333333333

#4. DecisionTreeClassifier
# model_score :  0.9333333333333333
# accuracy_score :  0.9333333333333333

#5. RandomForestClassifier
# model_score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

######## StandardScaler ###############################################
# 여기도 Linear 빼고 다 고만고만 Linear는 0.9에서 1.0 와리가리가 심해보임!

#1. LinearSVC
# model_score :  0.9333333333333333
# accuracy_score :  0.9333333333333333

#2. SVC
# model_score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

#3. KNeighboursClassifier
# model_score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

#4. DecisionTreeClassifier
# model_score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

#5. RandomForestClassifier
# model_score :  0.9666666666666667
# accuracy_score :  0.9666666666666667