import numpy as np

from sklearn.datasets import load_iris
data = load_iris()
y = data.target

from sklearn.model_selection import train_test_split as tts

y_train,y_test = tts(y,train_size = 0.8)
y_test = y_test.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y_test)
y_test = enc.transform(y_test).toarray()

print(y_test)