import glob
import os
import numpy as np
from PIL import Image

multi = list(range(12)) + list(range(24, 36))
single = list(range(12, 24)) + list(range(36, 48))

for i in range(1000):
    os.mkdir(f'../data/lpd/train2/{i}')
    os.mkdir(f'../data/lpd/train2/{i+1000}')

    for img in multi:
        multi_img = Image.open(f'../data/lpd/train/{i}/{img}.jpg')
        multi_img.save(f'../data/lpd/train2/{i}/{img}.jpg')

    for img in single:
        single_img = Image.open(f'../data/lpd/train/{i}/{img}.jpg')
        single_img.save(f'../data/lpd/train2/{i+1000}/{img}.jpg')