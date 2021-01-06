import numpy as np

datasets = np.array(range(1,11))
size = 7

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(datasets,size)
print("===========================")
print(dataset)
print(dataset.shape)

