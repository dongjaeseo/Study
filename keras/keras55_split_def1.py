# x len y len

# 1. 다:1 
# 2. 다:다
# 3. 다입력, 다:1
# 4. 다입력, 다:다 (열)
# 5. 다입력, 다:다 (행)

import numpy as np


# 1,2 다:다
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, timestep, ysize):
    x,y = [], []
    for i in range(len(dataset)):
        x_end = i + timestep
        y_end = x_end + ysize
        if y_end>len(dataset):
            break
        x_tmp = dataset[i:x_end]
        y_tmp = dataset[x_end:y_end]
        x.append(x_tmp)
        y.append(y_tmp)
    x = np.array(x)
    y = np.array(y)
    return(x,y)

x,y = split_xy1(dataset,4,2)
print(x)
print(y) 