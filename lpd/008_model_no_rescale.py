import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#0. 변수
batch = 16
seed = 42
dropout = 0.3
epochs = 1000
model_path = '../data/model/lpd_008.hdf5'
save_path = '../data/lpd/sample_008.csv'
sub = pd.read_csv('../data/lpd/sample.csv', header = 0)
es = EarlyStopping(patience = 7)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    preprocessing_function= preprocess_input,
    rotation_range= 20,
    brightness_range= [0.8, 1.1],
    zoom_range= 0.15,
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    '../data/lpd/train_new',
    target_size = (224, 224),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    '../data/lpd/train_new',
    target_size = (224, 224),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    '../data/lpd/test_new',
    target_size = (224, 224),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)

#2. 모델
eff = EfficientNetB4(include_top = False, input_shape=(224, 224, 3))
eff.trainable = False

pretrained = eff.output
globalpooling = GlobalAveragePooling2D()(pretrained)
layer_1 = Dense(2048)(globalpooling)
layer_1 = swish(layer_1)
layer_1 = Dropout(dropout)(layer_1)
layer_2 = Dense(1024)(layer_1)
layer_2 = swish(layer_2)
output = Dense(1000, activation = 'softmax')(layer_2)

model = Model(inputs = eff.input, outputs = output)

#3. 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['sparse_categorical_accuracy'])
model.fit(train_data, steps_per_epoch = np.ceil(39000/batch), validation_data= val_data, validation_steps= np.ceil(9000/batch),\
    epochs = epochs, callbacks = [es, cp, lr])

model = load_model(model_path)

#4. 평가 예측
pred = model.predict(test_data, steps = len(test_data))
pred = np.argmax(pred, 1)
print(pred)
print(len(pred))
sub.loc[:, 'prediction'] = pred
sub.to_csv(save_path, index = False)

# val_loss 0.27 정도? 