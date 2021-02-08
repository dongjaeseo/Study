import numpy as np
import pandas as pd
import PIL
from numpy import asarray
from PIL import Image

img=[]
for i in range(1):
    filepath='../data_2/dirty_mnist/%05d.png'%i
    image=Image.open(filepath)
    image_data=asarray(image)
    img.append(image_data)

img= np.array(img).reshape(256,256)
df = pd.DataFrame(img)
df.to_csv('../data_2/111.csv')