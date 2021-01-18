import numpy as np
import pandas as pd

k = pd.read_csv('./practice/dacon/data/hahaha.csv',index_col =0)

for i in range(81):
    y = []
    for j in range(2):
        y.append(k.iloc[[i],48*j:48*(j+1)].quantile(0.1))
    
y = np.array(y)