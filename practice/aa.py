import numpy as np
import pandas as pd

a = np.array(range(100))
a = a.reshape(2,2,5,5)
b = pd.DataFrame(a[0,0,1])
print(b)
