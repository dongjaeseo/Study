import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8)

model = Pipeline([('scale', MinMaxScaler()), ('model', RandomForestClassifier())])

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print("score : ", score)

# score :  0.9649122807017544