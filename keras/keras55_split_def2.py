# y의 가로 세로

# 1. 다:1 
# 2. 다:다
# 3. 다입력, 다:1
# 4. 다입력, 다:다 (열)
# 5. 다입력, 다:다 (행)

import numpy as np

# 3,4 다입력 다:다
# x = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
# dataset = np.transpose(x)

# def split_xy(dataset, timestep, ynum):
#     x, y = [], []
#     for i in range(len(dataset)):
#         x_end = i + timestep
#         y_end = x_end + ynum
#         if y_end>len(dataset):
#             break
#         x_tmp = dataset[i:x_end,:-(ynum)]
#         y_tmp = dataset[x_end:y_end,-ynum:]
#         x.append(x_tmp)
#         y.append(y_tmp)
#     x = np.array(x)
#     y = np.array(y)
#     return(x,y)

# x,y = split_xy(dataset,4,2)
# print(x)
# print(y)

# 5. 다입력 다:다 (행)
# x = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
# dataset = np.transpose(x)

# def split_xy(dataset,timestep,ynum):
#     x,y = [],[]
#     for i in range(len(dataset)):
#         x_end = i + timestep
#         y_end = x_end + ynum
#         if y_end>len(dataset):
#             break
#         x_tmp = dataset[i:x_end,:]
#         y_tmp = dataset[x_end:y_end,:]
#         x.append(x_tmp)
#         y.append(y_tmp)
#     x = np.array(x)
#     y = np.array(y)
#     return(x,y)

# x,y = split_xy(dataset,4,2)
# print(x)
# print(y)

# 4개 input?
x = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(x)

def split_xy(dataset,x_row,x_col,y_row,y_col):
    x,y = [],[]
    for i in range(len(dataset)):
        x_end = i + x_row
        y_end = x_end + y_row
        if y_end>len(dataset):
            break
        x_tmp = dataset[i:x_end,:x_col]
        y_tmp = dataset[x_end:y_end,-(y_col):]
        x.append(x_tmp)
        y.append(y_tmp)
    x = np.array(x)
    y = np.array(y)
    return(x,y)

x,y = split_xy(dataset,3,1,3,2)
print(x)
print(y)