import numpy as np
import pandas as pd

train = pd.read_csv('./practice/dacon/data/train/train.csv')
submission = pd.read_csv('./practice/dacon/data/sample_submission.csv')

def split_to_seq(data):
    tmp = pd.DataFrame()
    for i in range(int(len(data)/48)):
        for j in range(48):
            