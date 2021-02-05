import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# b = pd.read_csv('./mnist_1/alphabetb.csv', index_col = 0, header = 0)

b = pd.read_csv('./mnist_1/train.csv')
x = b.loc[:, '0':].to_numpy().reshape(-1,28,28)


# for j in range(len(x)):
for j in range(100,120):
    b = x[j]
    b = pd.DataFrame(b)

    b[b<30] = 0

    plt.imshow(b)
    plt.show()

    b = b.to_numpy().reshape(784,)
    
    number = 190
    patience = 150
    n= 0
    c = np.zeros((784,))
    c = b
    for i in range(len(b)):

        if b[i] < number:
            continue
        
        if b[i-1] >=40 and b[i-1] < number:
            if (b[i]-b[i-2]) >= patience:
                c[i-1] = b[i]


        if b[i+1] >=40 and b[i+1] < number:
                if (b[i]-b[i+2]) >= patience:
                    c[i+1] = b[i]
        
        
        if b[i-28] >=40 and b[i-28] < number:
            if (b[i]-b[i-56]) >= patience:
                c[i-28] = b[i]


        if b[i+28] >=40 and b[i+28] < number:
                if (b[i]-b[i+56]) >= patience:
                    c[i+28] = b[i]
        

    b = c.reshape(28,28)


    b = pd.DataFrame(b)
    plt.imshow(b)
    plt.show()
    # print(df)
