import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_diabetes()
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
models = [KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), LinearRegression()]

for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    # print(f'\n{i}')
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    print('r2_score : ', r2)

# KNeighborsRegressor()
# model_score :  0.37630871530410015
# r2_score :  0.37630871530410015   
# model_score :  0.48587733622645013  
# r2_score :  0.48587733622645013

# DecisionTreeRegressor()
# model_score :  -0.100617279118278 
# r2_score :  -0.100617279118278  
# model_score :  -0.022078086649546247
# r2_score :  -0.022078086649546247  

# RandomForestRegressor()
# model_score :  0.5143616159997054
# r2_score :  0.5143616159997054  
# model_score :  0.5075027359542723
# r2_score :  0.5075027359542723 

# LinearRegression()
# model_score :  0.5726954286533412
# r2_score :  0.5726954286533412
# model_score :  0.579850513720686
# r2_score :  0.579850513720686