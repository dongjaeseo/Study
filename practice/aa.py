import numpy as np
import pandas as pd

submission = pd.read_csv('./practice/dacon/data/test/0.csv')

from sklearn.preprocessing import StandardScaler
print(type(submission))
scale = StandardScaler()
print(type(submission))
scale.fit(submission[:2])
print(type(submission))
submission[:2] = scale.transform(submission[:2])
print(type(submission))
