import glob
import os
import numpy as np
from PIL import Image

multi = [range(9), range(10, 12), range(24, 36)]
single = [9, range(12, 24), range(36, 48)]

for i in range(3):
    os.mkdir('../data/lpd/train2/{i}')
    os.mkdir('../data/lpd/train2/{i+1000}')

    for img in multi:
        multi_img = Image.open('../data/lpd/train/{i}/{img}.jpg')
        multi_img.save('../data/lpd/train2/{i}/{img}.jpg')
        
    for img in single:
        single_img = Image.open('../data/lpd/train/{i}/{img}.jpg')
        single_img.save('../data/lpd/train2/{i+1000}/{img}.jpg')