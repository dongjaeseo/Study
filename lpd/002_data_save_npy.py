import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# 데이터
batch = 100000
seed = 42

train_gen = ImageDataGenerator(
    validation_split = 0.2,
    rescale = 1/255.
)

test_gen = ImageDataGenerator(
    rescale = 1/255.
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    '../data/lpd/train',
    target_size = (256, 256),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    '../data/lpd/train',
    target_size = (256, 256),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    '../data/lpd/test',
    target_size = (256, 256),
    class_mode = None,
    batch_size = batch,
    seed = seed,
    shuffle = False
)

# np.save('../data/npy/lpd_x_train_256.npy', arr=train_data[0][0])
# np.save('../data/npy/lpd_y_train_256.npy', arr=train_data[0][1])
np.save('../data/npy/lpd_x_val_256.npy', arr=val_data[0][0])
np.save('../data/npy/lpd_y_val_256.npy', arr=val_data[0][1])
np.save('../data/npy/lpd_x_test_256.npy', arr=test_data[0][0])
np.save('../data/npy/lpd_y_test_256.npy', arr=test_data[0][1])
