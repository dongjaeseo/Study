import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

b = pd.read_csv('./mnist_1/train.csv')
k = pd.read_csv('./mnist_1/test.csv')

x = b.loc[:, '0':].to_numpy().reshape(-1,28,28)
y = k.loc[:, '0':].to_numpy().reshape(-1,28,28)

def modif(x):
    d = []
    for j in range(len(x)):
        b = x[j]
        b = pd.DataFrame(b)

        b[b<30] = 0

        b = b.to_numpy().reshape(784,)
        
        number = 190
        patience = 150
        n= 0
        c = np.zeros((784,))
        c = b
        for i in range(len(b)):

            if b[i] < number:
                continue
            
            if b[i-1] >=60 and b[i-1] < number:
                if (b[i]-b[i-2]) >= patience:
                    c[i-1] = b[i]


            if b[i+1] >=60 and b[i+1] < number:
                    if (b[i]-b[i+2]) >= patience:
                        c[i+1] = b[i]
            
            
            if b[i-28] >=60 and b[i-28] < number:
                if (b[i]-b[i-56]) >= patience:
                    c[i-28] = b[i]


            if b[i+28] >=60 and b[i+28] < number:
                    if (b[i]-b[i+56]) >= patience:
                        c[i+28] = b[i]
            
        d.append(c)
    
    d = np.array(d)
    return d

x = modif(x)
y = modif(y)

b.loc[:, '0':] = x
b.to_csv('./mnist_1/train_new.csv', index = 0)

k.loc[:, '0':] = y
k.to_csv('./mnist_1/test_new.csv', index = 0)
