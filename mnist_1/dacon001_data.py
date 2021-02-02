import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./mnist_1/train.csv')
test = pd.read_csv('./mnist_1/test.csv')
print(train.shape, test.shape)