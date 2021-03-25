import numpy as np
import pandas as pd
import os, shutil
from PIL import Image


df = pd.read_csv('../data/lpd/submit_026/sample_026_09_percent.csv')
print(df.describe)
a = df.quantile(q = 0.21)
print(a)

aaa = df.to_numpy()
print(aaa.shape)
print(aaa)

# threshold = 0.763331
threshold = 0.4

i = 0
count_list = []
for percent in aaa:
    if percent < threshold:
        count_list.append(i)
    i+= 1

count_list = np.array(count_list)

np.save('../data/lpd/below_threshold.npy', count_list)

folder = '../data/lpd/test_new3/test_new3'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for img_idx in count_list:
    image = Image.open('../data/lpd/test_new/test_new/{0:05}.jpg'.format(img_idx))
    image.save('../data/lpd/test_new3/test_new3/{0:05}.jpg'.format(img_idx))
