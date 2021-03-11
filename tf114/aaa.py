import numpy as np

y = [[0.8, 0.1, 0.1],
     [0.6, 0.2, 0.2],
     [0.3, 0.5, 0.2]]

print(np.where([a == np.max(a) for a in y], 1, 0))