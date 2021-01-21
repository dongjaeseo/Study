import numpy as np
import pandas as pd

p = pd.DataFrame([[1,2,3],[4,5,6]])

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(p.loc[:,0])