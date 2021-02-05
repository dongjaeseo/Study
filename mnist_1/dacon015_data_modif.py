import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

b = pd.read_csv('./mnist_1/alphabetb.csv', index_col = 0, header = 0)





b[b<40] = 0

plt.imshow(b)
plt.show()

b = b.to_numpy().reshape(784,)
print(b)

patience = 180
n= 0
for i in range(len(b)):
    if b[i] < patience:
        continue
    
    if b[i-1] < patience:# and b[i-1] > 60:
        if (b[i]-b[i-2]) > patience:
            b[i-1] = b[i]

    if b[i+1] < patience:# and b[i+1] > 60:
        if b[i] != b[i-1]:
            if (b[i]-b[i+2]) > patience:
                b[i+1] = b[i]
    
    if b[i-28] < patience:# and b[i-28] > 60:
        if (b[i]-b[i-56]) > patience:
            b[i-28] = b[i]

    if b[i+28] < patience:# and b[i+28] > 60:
        if b[i] != b[i-28]:
            if (b[i]-b[i+56]) > patience:
                b[i+28] = b[i]

b = b.reshape(28,28)


b = pd.DataFrame(b)
plt.imshow(b)
plt.show()
# print(df)
