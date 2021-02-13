import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in range(50000,55000):

    image_path = f'../dacon/dirty/test_dirty_mnist/{str(i).zfill(5)}.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #254보다 작고 0이아니면 0으로 만들어주기
    image2 = np.where((image <= 254) & (image != 0), 0, image)
    image2 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    #dilate -> 이미지 팽창
    image2 = cv2.medianBlur(src=image2, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
    #medianBlur->커널 내의 필터중 밝기를 줄세워서 중간에 있는 값으로 현재 픽셀 값을 대체
    # image2 = cv2.erode(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    cv2.imwrite(f'../dacon/clean/test_clean_mnist/{str(i).zfill(5)}.png', image2)