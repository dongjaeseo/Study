import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['a','b'])
pred = np.array([1,2,3,4,5])
df['a'] = pred

print(df)