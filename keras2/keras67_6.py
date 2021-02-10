# '../data/image/gender/female, male' <<

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import matplotlib.pyplot as plt

test_datagen = ImageDataGenerator(
    rescale= 1/255.
)

batch = 16

x_test = test_datagen.flow_from_directory(
    '../data/image/n',
    class_mode = None,
    batch_size = batch,
    target_size =(128,128),
    shuffle= False
)
x_test.reset()

#2. 모델
model = load_model('../data/modelcheckpoint/gender_classification.hdf5')

#4. 평가
pred = model.predict_generator(x_test, steps = len(x_test[0][0])/batch)
for i in enumerate(pred):
    if i[1] <0.5:
        print(f'{i[0]+1}번째는 {np.round(((1-i[1])*100),2)}% 확률로 여자!')
    else:
        print(f'{i[0]+1}번째는 {np.round((i[1]*100),2)}% 확률로 남자!')