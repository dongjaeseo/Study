import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline # 둘이 거의 동일 쓰는법만 다르다

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header = 0, index_col = 0)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test, y_train,y_test = train_test_split(x,y,train_size = 0.8)

# from sklearn.preprocessing import MinMaxScaler
# scale = MinMaxScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)   # 이 부분을 파이프에 붙인거!!

# 모델     pipe : 전처리까지 합친다
# model = Pipeline([("scale", MinMaxScaler()),("model", RandomForestClassifier())])
model = make_pipeline(MinMaxScaler(),RandomForestClassifier())
model.fit(x_train,y_train)
result = model.score(x_test,y_test)
print(result)