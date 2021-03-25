import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# 이미지 검은배경 없애기
'''
for i in range(1000):

    for img in range(48,72):
        original = plt.imread('../data/lpd/train_new3/{0:04}/{1:02}.jpg'.format(i, img))
        original = np.array(original)

        original = original[20:204, 20:235, :]
        plt.imsave('../data/lpd/train_new3/{0:04}/{1:02}.jpg'.format(i, img), original)
'''
'''
# 이미지 리사이즈
for i in range(1000):
    
    for img in range(48,72):
        original = plt.imread('../data/lpd/train_new3/{0:04}/{1:02}.jpg'.format(i, img))
        original = np.array(original)
        original = image_resize(original, width = 256)
        plt.imsave('../data/lpd/train_new3/{0:04}/{1:02}.jpg'.format(i, img), original)
'''
# 이미지 보더

for i in range(1000):
    for img in range(48,72):
        original = cv2.imread('../data/lpd/train_new3/{0:04}/{1:02}.jpg'.format(i, img)) 
        original2 = cv2.copyMakeBorder(original, 18, 19, 0, 0, cv2.BORDER_REPLICATE)
        cv2.imwrite('../data/lpd/train_new3/{0:04}/{1:02}.jpg'.format(i, img), original2)


# csv 로 위치 확인
'''
original = pd.DataFrame(original[:,:,0])
original.to_csv('../data/lpd/aaa.csv', index = False)
'''
