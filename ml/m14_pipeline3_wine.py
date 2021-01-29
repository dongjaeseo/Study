import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size= 0.8)

scales = [MinMaxScaler(), StandardScaler()]
for i in scales:
    model = Pipeline([('scale', i), ('model', RandomForestClassifier())])

    model.fit(x_train,y_train)

    score = model.score(x_test, y_test)
    print(f'score_{i} : ', score)

# score_MinMaxScaler() :  0.9722222222222222
# score_StandardScaler() :  0.9722222222222222