import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt

#0. 변수
batch = 16
seed = 42
dropout = 0.3
epochs = 1000
model_path = '../data/model/lpd_001.hdf5'
sub = pd.read_csv('../data/lpd/sample.csv', header = 0)
es = EarlyStopping(patience = 5)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    preprocessing_function= preprocess_input
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    rescale = 1/255.
)

# # Found 39000 images belonging to 1000 classes.
# train_data = train_gen.flow_from_directory(
#     '../data/lpd/train',
#     target_size = (128, 128),
#     class_mode = 'sparse',
#     batch_size = batch,
#     seed = seed,
#     subset = 'training'
# )

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    '../data/lpd/train',
    target_size = (128, 128),
    class_mode = 'sparse',
    batch_size = batch,
    shuffle = False,
    subset = 'validation'
)

test_data = test_gen.flow_from_directory(
    '../data/lpd/test',
    target_size = (128, 128),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)

plt.imshow(test_data[0][2])
plt.show()