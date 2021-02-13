import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in range(50000):

    image_path = f'../dacon/dirty/dirty_mnist/{str(i).zfill(5)}.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #254보다 작고 0이아니면 0으로 만들어주기
    image2 = np.where((image <= 254) & (image != 0), 0, image)
    image2 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    #dilate -> 이미지 팽창
    image2 = cv2.medianBlur(src=image2, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
    #medianBlur->커널 내의 필터중 밝기를 줄세워서 중간에 있는 값으로 현재 픽셀 값을 대체
    # image2 = cv2.erode(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    cv2.imwrite(f'../dacon/clean/clean_mnist/{str(i).zfill(5)}.png', image2)

'''
img1 = cv2.imread('../dacon/dirty/dirty_mnist/00000.png')
img1 = np.asarray(img1)
print(img1.shape)
cv2.imshow('original',img1)

img2 = cv2.imread('../dacon/dirty/dirty_mnist/00000.png', cv2.IMREAD_COLOR)
img2 = np.asarray(img2)
print(img2.shape)
cv2.imshow('color',img2)

img3 = cv2.imread('../dacon/dirty/dirty_mnist/00000.png', cv2.IMREAD_GRAYSCALE)
img3 = np.asarray(img3)
print(img3.shape)
cv2.imshow('grey',img3)
cv2.waitKey(0)

# (256, 256, 3)
# (256, 256, 3)
# (256, 256) 
'''