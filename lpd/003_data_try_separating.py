import glob
import os
import numpy as np
from PIL import Image

multi = list(range(12)) + list(range(24, 36))
single = list(range(12, 24)) + list(range(36, 48))

# 코드 실행 시 단품과 세트를 구분해서 다른 라벨링을 해준다!
for i in range(1000):
    os.mkdir('../data/lpd/train2/{0:04}'.format(i))
    os.mkdir('../data/lpd/train2/{0:04}'.format(i+1000))

    for img in multi:
        multi_img = Image.open(f'../data/lpd/train/{i}/{img}.jpg')
        multi_img.save('../data/lpd/train2/{0:04}/{1:02}.jpg'.format(i, img))

    for img in single:
        single_img = Image.open(f'../data/lpd/train/{i}/{img}.jpg')
        single_img.save('../data/lpd/train2/{0:04}/{1:02}.jpg'.format(i+1000, img))

